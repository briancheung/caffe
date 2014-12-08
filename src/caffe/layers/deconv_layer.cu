// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void DeconvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int bottom_offset = M_ * N_;
  for (int n = 0; n < num_; ++n) {
    // First, innerproduct with groups
    for (int g = 0; g < group_; ++g) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
          Dtype(1), weight + weight_offset * g,
          bottom_data + bottom[0]->offset(n) + bottom_offset * g,
          Dtype(0), col_data + col_offset * g);
    }
    // Second, add bias
    if (bias_term_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
          N_, 1, Dtype(1), this->blobs_[1]->gpu_data(),
          bias_multiplier_.gpu_data(), Dtype(1), col_data);
    }
    // Finally, col2im
    col2im_gpu(col_data, num_output_, height_out_, width_out_,
        kernel_h_, kernel_w_, pad_h_, pad_w_,
        stride_h_, stride_w_, top_data + top[0]->offset(n));
  }
}

template <typename Dtype>
void DeconvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0] || this->param_propagate_down_[0] ||
      (bias_term_ && this->param_propagate_down_[1])) {
    int weight_offset = M_ * K_;
    int top_offset = K_ * N_;
    int bottom_offset = M_ * N_;
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* col_data = col_buffer_.mutable_gpu_data();
    Dtype* weight_diff;
    if (this->param_propagate_down_[0]) {
      weight_diff = this->blobs_[0]->mutable_gpu_diff();
      caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
    }
    Dtype* bias_diff;
    if (bias_term_ && this->param_propagate_down_[1]) {
      bias_diff = this->blobs_[1]->mutable_gpu_diff();
      caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
    }
    for (int n = 0; n < num_; ++n) {
      im2col_gpu(top_diff + top[0]->offset(n), num_output_,
          height_out_, width_out_, kernel_h_, kernel_w_, pad_h_, pad_w_,
          stride_h_, stride_w_, col_data);
      // gradient w.r.t. weight. Note that we will accumulate diffs.
      if (this->param_propagate_down_[0]) {
        for (int g = 0; g < group_; ++g) {
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
            Dtype(1), bottom_data + bottom[0]->offset(n) + bottom_offset * g,
            col_data + top_offset * g,
            Dtype(1), weight_diff + weight_offset * g);
        }
      }
      if (bias_term_ && this->param_propagate_down_[1]) {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, N_, Dtype(1), col_data,
            bias_multiplier_.gpu_data(), Dtype(1), bias_diff);
      }
      // gradient w.r.t. bottom data, if necessary
      if (propagate_down[0]) {
        const Dtype* weight = this->blobs_[0]->gpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        for (int g = 0; g < group_; ++g) {
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
            Dtype(1), weight + weight_offset * g,
            col_data + top_offset * g, Dtype(0),
            bottom_diff + bottom[0]->offset(n) + bottom_offset * g);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DeconvolutionLayer);


}  // namespace caffe

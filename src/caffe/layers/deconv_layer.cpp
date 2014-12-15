// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void DeconvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  kernel_h_ = this->layer_param_.convolution_param().kernel_size();
  kernel_w_ = kernel_h_;
  stride_h_ = this->layer_param_.convolution_param().stride();
  stride_w_ = stride_h_;
  pad_h_ = this->layer_param_.convolution_param().pad();
  pad_w_ = pad_h_;
  group_ = this->layer_param_.convolution_param().group();
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  // The im2col result buffer holds only one image at a time to avoid
  // overly large memory usage.
  height_out_ = (height_ - 1) * stride_h_ + kernel_h_ - 2 * pad_h_;
  width_out_ = (width_ - 1) * stride_w_ + kernel_w_ - 2 * pad_w_;
  col_buffer_.Reshape(
      1, num_output_ * kernel_h_ * kernel_w_, height_, width_);
  // Set the parameters
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  // Figure out the dimensions for individual gemms.
  M_ = channels_ / group_;
  K_ = num_output_ * kernel_h_ * kernel_w_ / group_;
  N_ = height_ * width_;
  top[0]->Reshape(bottom[0]->num(), num_output_, height_out_, width_out_);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
      this->set_param_propagate_down(1, true);
    } else {
      this->blobs_.resize(1);
    }
    this->set_param_propagate_down(0, true);
    // Intialize the weight
    this->blobs_[0].reset(new Blob<Dtype>(
        channels_, num_output_ / group_, kernel_h_, kernel_w_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, num_output_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Set up the bias filler
  if (bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1, N_);
    caffe_set(N_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}


template <typename Dtype>
void DeconvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
/*
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
    " convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
        << "Inputs must have same width.";
  }
  // Shape the tops.
  height_out_ =
      (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  width_out_ = (width_ + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }
  // Prepare the matrix multiplication computation.
  // Each input will be convolved as a single GEMM.
  M_ = num_output_ / group_;
  K_ = channels_ * kernel_h_ * kernel_w_ / group_;
  N_ = height_out_ * width_out_;
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_.Reshape(
      1, channels_ * kernel_h_ * kernel_w_, height_out_, width_out_);
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1, N_);
    caffe_set(N_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
*/
}



template <typename Dtype>
void DeconvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* col_data = col_buffer_.mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int bottom_offset = M_ * N_;
  for (int n = 0; n < num_; ++n) {
    // First, innerproduct with groups
    for (int g = 0; g < group_; ++g) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
        Dtype(1), weight + weight_offset * g,
        bottom_data + bottom[0]->offset(n) + bottom_offset * g,
        Dtype(0), col_data + col_offset * g);
    }
    // Second, add bias
    if (bias_term_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
          N_, 1, Dtype(1), this->blobs_[1]->cpu_data(),
          bias_multiplier_.cpu_data(), Dtype(1), col_data);
    }
    // Finally, col2im
    col2im_cpu(col_data, num_output_, height_out_, width_out_,
        kernel_h_, kernel_w_, pad_h_, pad_w_,
        stride_h_, stride_w_, top_data + top[0]->offset(n));
  }
}

template <typename Dtype>
void DeconvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0] || this->param_propagate_down_[0] ||
      (bias_term_ && this->param_propagate_down_[1])) {
    int weight_offset = M_ * K_;
    int top_offset = K_ * N_;
    int bottom_offset = M_ * N_;
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* col_data = col_buffer_.mutable_cpu_data();
    Dtype* weight_diff = NULL;
    if (this->param_propagate_down_[0]) {
      weight_diff = this->blobs_[0]->mutable_cpu_diff();
      caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
    }
    Dtype* bias_diff = NULL;
    if (bias_term_ && this->param_propagate_down_[1]) {
      bias_diff = this->blobs_[1]->mutable_cpu_diff();
      caffe_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
    }
    for (int n = 0; n < num_; ++n) {
      im2col_cpu(top_diff + top[0]->offset(n), num_output_,
          height_out_, width_out_, kernel_h_, kernel_w_, pad_h_, pad_w_,
          stride_h_, stride_w_, col_data);
      // gradient w.r.t. weight. Note that we will accumulate diffs.
      if (this->param_propagate_down_[0]) {
        for (int g = 0; g < group_; ++g) {
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
            Dtype(1), bottom_data + bottom[0]->offset(n) + bottom_offset * g,
            col_data + top_offset * g,
            Dtype(1), weight_diff + weight_offset * g);
        }
      }
      if (bias_term_ && this->param_propagate_down_[1]) {
        caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, N_, Dtype(1), col_data,
            bias_multiplier_.cpu_data(), Dtype(1),
            bias_diff);
      }
      // gradient w.r.t. bottom data, if necessary
      if (propagate_down[0]) {
        const Dtype* weight = this->blobs_[0]->cpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        for (int g = 0; g < group_; ++g) {
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
            Dtype(1), weight + weight_offset * g,
            col_data + top_offset * g, Dtype(0),
            bottom_diff + bottom[0]->offset(n) + bottom_offset * g);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(DeconvolutionLayer);
#endif

INSTANTIATE_CLASS(DeconvolutionLayer);

}  // namespace caffe

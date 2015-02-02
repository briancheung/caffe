#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void XCovLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  mean_vec_.clear();
  mean_vec_.push_back(&mean_0_);
  mean_vec_.push_back(&mean_1_);

  temp_vec_.clear();
  temp_vec_.push_back(&temp_0_);
  temp_vec_.push_back(&temp_1_);
}


template <typename Dtype>
void XCovLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());

  for (int i = 0 ; i < bottom.size() ; i++ ) {
    mean_vec_[i]->Reshape(1, bottom[i]->channels(),
        bottom[i]->height(), bottom[i]->width());
    temp_vec_[i]->Reshape(bottom[i]->num(), bottom[i]->channels(),
        bottom[i]->height(), bottom[i]->width());
  }
  sum_multiplier_.Reshape(bottom[0]->num(), 1, 1, 1);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
}

template <typename Dtype>
void XCovLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();

  // for now, we support only two inputs
  CHECK_EQ(bottom.size(), 2);

  for (int i = 0 ; i < bottom.size() ; i++) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    int num = bottom[i]->num();
    int dim = bottom[i]->count() / num;

    // calculate mean vector over batch
    caffe_cpu_gemv<Dtype>(CblasTrans, num, dim, 1. / num, bottom_data,
        sum_multiplier_.cpu_data(), 0., mean_vec_[i]->mutable_cpu_data());

    // broadcast and negative mean vector
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
        sum_multiplier_.cpu_data(),
        mean_vec_[i]->cpu_data(),
        0.,
        temp_vec_[i]->mutable_cpu_data());

    // subtract mean
    caffe_add(temp_vec_[i]->count(), bottom_data, temp_vec_[i]->cpu_data(),
        temp_vec_[i]->mutable_cpu_data());
  }

  //top_data = temp_[0].T * temp_[1].T;

  int num = bottom[0]->num();
  int dim0 = bottom[0]->count() / num;
  int dim1 = bottom[1]->count() / num;

  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, dim0, dim1, num, 1.,
      temp_vec_[0]->cpu_data(),
      temp_vec_[1]->cpu_data(),
      0.,
      top_data);
}

template <typename Dtype>
void XCovLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  for (int i = 0 ; i < bottom.size() ; i++) {
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / num;

    // calculate mean vector over batch
    caffe_cpu_gemv<Dtype>(CblasTrans, num, dim, 1. / num, top_diff,
        sum_multiplier_.cpu_data(), 0., mean_vec_[i]->mutable_cpu_data());

    // broadcast and negative mean vector
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
            sum_multiplier_.cpu_data(),
            mean_vec_[i]->cpu_data(),
            0.,
            temp_vec_[i]->mutable_cpu_data());

    // subtract mean
    caffe_add(temp_vec_[i]->count(), top_diff, temp_vec_[i]->cpu_data(), bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(XCovLossLayer);
#endif

INSTANTIATE_CLASS(XCovLossLayer);


}  // namespace caffe

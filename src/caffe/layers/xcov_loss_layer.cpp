#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void XCovLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  mean_.Reshape(1, bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  temp_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  sum_multiplier_.Reshape(bottom[0]->num(), 1, 1, 1);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
}

template <typename Dtype>
void XCovLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;

  // calculate mean vector over batch
  caffe_cpu_gemv<Dtype>(CblasTrans, num, dim, 1. / num, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());

  // broadcast and negative mean vector
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
          sum_multiplier_.cpu_data(),
          mean_.cpu_data(),
          0.,
          temp_.mutable_cpu_data());

  // subtract mean
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);
}

template <typename Dtype>
void XCovLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;

  // calculate mean vector over batch
  caffe_cpu_gemv<Dtype>(CblasTrans, num, dim, 1. / num, top_diff,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());

  // broadcast and negative mean vector
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
          sum_multiplier_.cpu_data(),
          mean_.cpu_data(),
          0.,
          temp_.mutable_cpu_data());

  // subtract mean
  caffe_add(temp_.count(), top_diff, temp_.cpu_data(), bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(XCovLossLayer);
#endif

INSTANTIATE_CLASS(XCovLossLayer);


}  // namespace caffe

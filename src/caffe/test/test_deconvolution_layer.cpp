// Copyright 2014 BVLC and contributors.

#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename TypeParam>
class DeconvolutionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DeconvolutionLayerTest()
      : blob_bottom_conv_(new Blob<Dtype>()),
        blob_bottom_deconv_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_bottom_conv_->Reshape(2, 4, 3, 3);
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_conv_);
    blob_bottom_conv_vec_.push_back(blob_bottom_conv_);
    blob_bottom_deconv_vec_.push_back(blob_bottom_deconv_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~DeconvolutionLayerTest() {
    delete blob_bottom_conv_;
    delete blob_bottom_deconv_;
    delete blob_top_;
  }
  Blob<Dtype>* blob_bottom_conv_;
  Blob<Dtype>* const blob_bottom_deconv_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_conv_vec_;
  vector<Blob<Dtype>*> blob_bottom_deconv_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DeconvolutionLayerTest, TestDtypesAndDevices);

TYPED_TEST(DeconvolutionLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_conv_->Reshape(2, 6, 7, 7);
  LayerParameter conv_layer_param;
  ConvolutionParameter* convolution_param =
      conv_layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(4);
  shared_ptr<Layer<Dtype> > conv_layer(
      new ConvolutionLayer<Dtype>(conv_layer_param));
  conv_layer->SetUp(this->blob_bottom_conv_vec_,
                    this->blob_bottom_deconv_vec_);
  EXPECT_EQ(this->blob_bottom_deconv_->num(), 2);
  EXPECT_EQ(this->blob_bottom_deconv_->channels(), 4);
  EXPECT_EQ(this->blob_bottom_deconv_->height(), 3);
  EXPECT_EQ(this->blob_bottom_deconv_->width(), 3);
  EXPECT_EQ(conv_layer->blobs().size(), 2);
  EXPECT_EQ(conv_layer->blobs()[0]->num(), 4);
  EXPECT_EQ(conv_layer->blobs()[0]->channels(), 6);
  EXPECT_EQ(conv_layer->blobs()[0]->height(), 3);
  EXPECT_EQ(conv_layer->blobs()[0]->width(), 3);
  EXPECT_EQ(conv_layer->blobs()[1]->num(), 1);
  EXPECT_EQ(conv_layer->blobs()[1]->channels(), 1);
  EXPECT_EQ(conv_layer->blobs()[1]->height(), 1);
  EXPECT_EQ(conv_layer->blobs()[1]->width(), 4);

  LayerParameter deconv_layer_param;
  ConvolutionParameter* deconvolution_param =
      deconv_layer_param.mutable_convolution_param();
  deconvolution_param->set_kernel_size(3);
  deconvolution_param->set_stride(2);
  deconvolution_param->set_num_output(6);
  shared_ptr<Layer<Dtype> > layer(
      new DeconvolutionLayer<Dtype>(deconv_layer_param));
  layer->SetUp(this->blob_bottom_deconv_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 6);
  EXPECT_EQ(this->blob_top_->height(), 7);
  EXPECT_EQ(this->blob_top_->width(), 7);
  EXPECT_EQ(layer->blobs().size(), 2);
  EXPECT_EQ(layer->blobs()[0]->num(), 4);
  EXPECT_EQ(layer->blobs()[0]->channels(), 6);
  EXPECT_EQ(layer->blobs()[0]->height(), 3);
  EXPECT_EQ(layer->blobs()[0]->width(), 3);
  EXPECT_EQ(layer->blobs()[1]->num(), 1);
  EXPECT_EQ(layer->blobs()[1]->channels(), 1);
  EXPECT_EQ(layer->blobs()[1]->height(), 1);
  EXPECT_EQ(layer->blobs()[1]->width(), 6);
}

TYPED_TEST(DeconvolutionLayerTest, TestSetupPadded) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_conv_->Reshape(2, 6, 7, 7);
  LayerParameter conv_layer_param;
  ConvolutionParameter* convolution_param =
      conv_layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->set_pad(2);
  shared_ptr<Layer<Dtype> > conv_layer(
      new ConvolutionLayer<Dtype>(conv_layer_param));
  conv_layer->SetUp(this->blob_bottom_conv_vec_,
                    this->blob_bottom_deconv_vec_);
  EXPECT_EQ(this->blob_bottom_deconv_->num(), 2);
  EXPECT_EQ(this->blob_bottom_deconv_->channels(), 4);
  EXPECT_EQ(this->blob_bottom_deconv_->height(), 5);
  EXPECT_EQ(this->blob_bottom_deconv_->width(), 5);

  LayerParameter deconv_layer_param;
  ConvolutionParameter* deconvolution_param =
      deconv_layer_param.mutable_convolution_param();
  deconvolution_param->set_kernel_size(3);
  deconvolution_param->set_stride(2);
  deconvolution_param->set_num_output(6);
  deconvolution_param->set_pad(2);
  shared_ptr<Layer<Dtype> > layer(
      new DeconvolutionLayer<Dtype>(deconv_layer_param));
  layer->SetUp(this->blob_bottom_deconv_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 6);
  EXPECT_EQ(this->blob_top_->height(), 7);
  EXPECT_EQ(this->blob_top_->width(), 7);
  EXPECT_EQ(layer->blobs().size(), 2);
  EXPECT_EQ(layer->blobs()[0]->num(), 4);
  EXPECT_EQ(layer->blobs()[0]->channels(), 6);
  EXPECT_EQ(layer->blobs()[0]->height(), 3);
  EXPECT_EQ(layer->blobs()[0]->width(), 3);
  EXPECT_EQ(layer->blobs()[1]->num(), 1);
  EXPECT_EQ(layer->blobs()[1]->channels(), 1);
  EXPECT_EQ(layer->blobs()[1]->height(), 1);
  EXPECT_EQ(layer->blobs()[1]->width(), 6);
}

// Input: values are channel index + 1.
//     [ 1 1 1 1 1 ] [ 2 2 2 2 2 ] [ 3 3 3 3 3 ]
//     [ 1 1 1 1 1 ] [ 2 2 2 2 2 ] [ 3 3 3 3 3 ]
//     [ 1 1 1 1 1 ] [ 2 2 2 2 2 ] [ 3 3 3 3 3 ]
//     [ 1 1 1 1 1 ] [ 2 2 2 2 2 ] [ 3 3 3 3 3 ]
//     [ 1 1 1 1 1 ] [ 2 2 2 2 2 ] [ 3 3 3 3 3 ]
//
// Filters: positive in one channel each.
// Filter 0:
//     [  0   0   0  ]  [ 1/9 1/9 1/9 ]  [  0   0   0  ]
//     [  0   0   0  ]  [ 1/9 1/9 1/9 ]  [  0   0   0  ]
//     [  0   0   0  ]  [ 1/9 1/9 1/9 ]  [  0   0   0  ]
// Filter 1:
//     [  0   0   0  ]  [  0   0   0  ]  [ 1/9 1/9 1/9 ]
//     [  0   0   0  ]  [  0   0   0  ]  [ 1/9 1/9 1/9 ]
//     [  0   0   0  ]  [  0   0   0  ]  [ 1/9 1/9 1/9 ]
// Filter 2:
//     [  0   0   0  ]  [ 1/9 1/9 1/9 ]  [  0   0   0  ]
//     [  0   0   0  ]  [ 1/9 1/9 1/9 ]  [  0   0   0  ]
//     [  0   0   0  ]  [ 1/9 1/9 1/9 ]  [  0   0   0  ]
//
// Conv outputs / deconv inputs:
//     [ 2 2 ] [ 3 3 ] [ 2 2 ]
//     [ 2 2 ] [ 3 3 ] [ 2 2 ]
//
// Deconv outputs:
//     [ 0 0 0 0 0 ] [ 4/9 4/9  8/9 4/9 4/9 ] [ 1/3 1/3 2/3 1/3 1/3 ]
//     [ 0 0 0 0 0 ] [ 4/9 4/9  8/9 4/9 4/9 ] [ 1/3 1/3 2/3 1/3 1/3 ]
//     [ 0 0 0 0 0 ] [ 8/9 8/9 16/9 8/9 8/9 ] [ 2/3 2/3 4/3 2/3 2/3 ]
//     [ 0 0 0 0 0 ] [ 4/9 4/9  8/9 4/9 4/9 ] [ 1/3 1/3 2/3 1/3 1/3 ]
//     [ 0 0 0 0 0 ] [ 4/9 4/9  8/9 4/9 4/9 ] [ 1/3 1/3 2/3 1/3 1/3 ]
TYPED_TEST(DeconvolutionLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_conv_->Reshape(1, 3, 5, 5);
  LayerParameter conv_layer_param;
  ConvolutionParameter* convolution_param =
      conv_layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(3);
  convolution_param->set_bias_term(false);
  shared_ptr<Layer<Dtype> > conv_layer(
      new ConvolutionLayer<Dtype>(conv_layer_param));
  conv_layer->SetUp(this->blob_bottom_conv_vec_,
                    this->blob_bottom_deconv_vec_);
  EXPECT_EQ(this->blob_bottom_deconv_->num(), 1);
  EXPECT_EQ(this->blob_bottom_deconv_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_deconv_->height(), 2);
  EXPECT_EQ(this->blob_bottom_deconv_->width(), 2);

  LayerParameter deconv_layer_param;
  ConvolutionParameter* deconvolution_param =
      deconv_layer_param.mutable_convolution_param();
  deconvolution_param->set_kernel_size(3);
  deconvolution_param->set_stride(2);
  deconvolution_param->set_num_output(3);
  deconvolution_param->set_bias_term(false);
  shared_ptr<Layer<Dtype> > layer(
      new DeconvolutionLayer<Dtype>(deconv_layer_param));
  layer->SetUp(this->blob_bottom_deconv_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 5);

  Blob<Dtype>* conv_filters = conv_layer->blobs()[0].get();
  Blob<Dtype>* deconv_filters = layer->blobs()[0].get();
  ASSERT_EQ(conv_filters->num(), 3);
  ASSERT_EQ(conv_filters->channels(), 3);
  ASSERT_EQ(conv_filters->height(), 3);
  ASSERT_EQ(conv_filters->width(), 3);
  ASSERT_EQ(deconv_filters->num(), 3);
  ASSERT_EQ(deconv_filters->channels(), 3);
  ASSERT_EQ(deconv_filters->height(), 3);
  ASSERT_EQ(deconv_filters->width(), 3);
  Dtype* conv_data = conv_filters->mutable_cpu_data();
  Dtype* deconv_data = deconv_filters->mutable_cpu_data();
  const Dtype norm = conv_filters->height() * conv_filters->width();
  int offset = 0;
  for (int n = 0; n < conv_filters->num(); ++n) {
    for (int c = 0; c < conv_filters->channels(); ++c) {
      for (int h = 0; h < conv_filters->height(); ++h) {
        for (int w = 0; w < conv_filters->width(); ++w) {
          const Dtype value = ((n == 0 && c == 1) || (n == 1 && c == 2) ||
                               (n == 2 && c == 1)) / norm;
          conv_data[offset] = value;
          deconv_data[offset] = value;
          ++offset;
        }
      }
    }
  }
  // Conv inputs: values are channel index + 1.
  Dtype* input_data = this->blob_bottom_conv_->mutable_cpu_data();
  offset = 0;
  for (int n = 0; n < this->blob_bottom_conv_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_conv_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_conv_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_conv_->width(); ++w) {
          input_data[offset++] = c + 1;
        }
      }
    }
  }
  conv_layer->Forward(this->blob_bottom_conv_vec_,
                      this->blob_bottom_deconv_vec_);
  const Dtype* output_data = this->blob_bottom_deconv_->cpu_data();
  for (int i = 0; i < 8; ++i) {
    if (i < 4) {
      EXPECT_FLOAT_EQ(2, output_data[i]);
    } else if (i < 8) {
      EXPECT_FLOAT_EQ(3, output_data[i]);
    } else {
      EXPECT_FLOAT_EQ(2, output_data[i]);
    }
  }

  layer->Forward(this->blob_bottom_deconv_vec_, this->blob_top_vec_);
  output_data = this->blob_top_->cpu_data();
  for (int n = 0; n < this->blob_top_->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          Dtype expected_value = 0;
          if (c == 1) {
            if (h == 2 && w == 2) {
              expected_value = 16 / 9.0;
            } else if (h == 2 || w == 2) {
              expected_value = 8 / 9.0;
            } else {
              expected_value = 4 / 9.0;
            }
          } else if (c == 2) {
            if (h == 2 && w == 2) {
              expected_value = 4 / 3.0;
            } else if (h == 2 || w == 2) {
              expected_value = 2 / 3.0;
            } else {
              expected_value = 1 / 3.0;
            }
          }
          EXPECT_FLOAT_EQ(expected_value, this->blob_top_->data_at(n, c, h, w));
        }
      }
    }
  }
}

TYPED_TEST(DeconvolutionLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_group(2);
  convolution_param->set_pad(2);
  convolution_param->set_num_output(2);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > conv_layer(
      new ConvolutionLayer<Dtype>(layer_param));
  conv_layer->SetUp(this->blob_bottom_conv_vec_,
                    this->blob_bottom_deconv_vec_);
  conv_layer->Forward(this->blob_bottom_conv_vec_,
                      this->blob_bottom_deconv_vec_);
  convolution_param->set_num_output(4);
  DeconvolutionLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_deconv_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_bottom_deconv_->num(), 2);
  EXPECT_EQ(this->blob_bottom_deconv_->channels(), 2);
  EXPECT_EQ(this->blob_bottom_deconv_->height(), 3);
  EXPECT_EQ(this->blob_bottom_deconv_->width(), 3);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
  EXPECT_EQ(layer.blobs().size(), 2);
  EXPECT_EQ(layer.blobs()[0]->num(), 2);
  EXPECT_EQ(layer.blobs()[0]->channels(), 2);
  EXPECT_EQ(layer.blobs()[0]->height(), 3);
  EXPECT_EQ(layer.blobs()[0]->width(), 3);
  EXPECT_EQ(layer.blobs()[1]->num(), 1);
  EXPECT_EQ(layer.blobs()[1]->channels(), 1);
  EXPECT_EQ(layer.blobs()[1]->height(), 1);
  EXPECT_EQ(layer.blobs()[1]->width(), 4);

  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_deconv_vec_,
      this->blob_top_vec_);
}


}  // namespace caffe

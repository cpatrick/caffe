#ifndef CAFFE_MEAN_LAYER_HPP_
#define CAFFE_MEAN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Can normalize data by subtracting mean and scaling
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class MeanLayer : public Layer<Dtype> {
 public:
  explicit MeanLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~MeanLayer() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "Mean"; }

 protected:
  Blob<Dtype> mean_;
  void ReadData();

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  {
    LOG(FATAL) << "MeanLayer cannot do backward.";
  }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  {
    LOG(FATAL) << "MeanLayer cannot do backward.";
  }
};

}  // namespace caffe

#endif  // CAFFE_MEAN_LAYER_HPP_

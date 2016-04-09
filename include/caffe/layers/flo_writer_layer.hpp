#ifndef CAFFE_FLO_WRITER_LAYER_HPP_
#define CAFFE_FLO_WRITER_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief FLOWriterLayer writes FLO (flow) files
 *
 */
template <typename Dtype>
class FLOWriterLayer : public Layer<Dtype> {
 public:
  explicit FLOWriterLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~FLOWriterLayer() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "FLOWriter"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 0; }
  
 protected:

  void writeFloFile(string filename, const float* data, int xSize, int ySize);
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
};

}  // namespace caffe

#endif // CAFFE_FLO_WRITER_LAYER_HPP_


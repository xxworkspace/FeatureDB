
#include "FeatureDB.h"
#include "shared_lock.h"
#include "hnswlib/hnswalg.h"

namespace bigo{
namespace ml{

  static SharedLock slock;
  typedef hnswlib::HierarchicalNSW<float> INDEX;
  template<class T>
  FeatureDB<T>::FeatureDB(
    const std::string space_name,
    const unsigned dim,
    const unsigned M,
    const unsigned max_elements,
    const unsigned query_ef,
    const unsigned construction_ef,
    const std::string dtype)
    :Dim(dim) {
    dist = space_name;
    normalize = false;
    initialize = false;
    hnswlib::SpaceInterface<float> *space;
    if (space_name == "ip") space = new hnswlib::InnerProductSpace(dim);
    else if (space_name == "l2") space = new hnswlib::L2Space(dim);
    else if (space_name == "cosine") {
      space = new hnswlib::InnerProductSpace(dim);
      normalize = true;
    }

    hnsw = (void*)new INDEX(space, max_elements, M, construction_ef);
    ((INDEX*)hnsw)->setEf(query_ef);
  }

  template<class T>
  FeatureDB<T>::~FeatureDB() {
    delete (INDEX*)hnsw;
  }

  template<class T>
  void FeatureDB<T>::normalization(const T* src, T* dst,const unsigned dim) {
    const T* tmp = src;
    T sum = 0;
    for (int i = 0; i < dim; ++i) {
      sum += (*tmp)*(*tmp);
      ++tmp;
    }
    sum = sqrt(sum);
    for (int i = 0; i < dim; ++i)
      *(dst++) = *(src++) / sum;
  }

  template<class T>
  unsigned FeatureDB<T>::size() {
    return ((INDEX*)hnsw)->cur_element_count;
  }

  template<class T>
  const std::vector<char> FeatureDB<T>::dump() {
    std::vector<char> serial;
    ((INDEX*)hnsw)->saveIndex(serial);
    return serial;
  }

  template<class T>
  bool FeatureDB<T>::load(const std::vector<char>& serial) {
    if (initialize) return false;
    initialize = true;
    ((INDEX*)hnsw)->loadIndex(serial);
    return true;
  }

  template<class T>
  void FeatureDB<T>::save(const std::string filename){
    slock.wLock();
    ((INDEX*)hnsw)->saveIndex(filename);
    slock.wRelease();
  }

  template<class T>
  bool FeatureDB<T>::restore(const std::string filename){
    if(initialize) return false;
    delete ((INDEX*)hnsw);
    hnswlib::SpaceInterface<float> *space;
    if (dist == "ip") space = new hnswlib::InnerProductSpace(Dim);
    else if (dist == "l2") space = new hnswlib::L2Space(Dim);
    else if (dist == "cosine") {
      space = new hnswlib::InnerProductSpace(Dim);
      normalize = true;
    }
    hnsw = (void*)new INDEX(space,filename);

    initialize = true;
    return true;
  }
#define CHECK_DIM(obj, src, rt) \
    if (obj != src) return rt;

  template<class T>
  bool FeatureDB<T>::insert(const std::vector<T>& data,const uint64_t label) {
    initialize = true;
    CHECK_DIM(data.size(), Dim, false)
    if (normalize) {
      float *tmp = new float[Dim];
      normalization(&data[0], tmp, Dim);
      slock.rLock();
      ((INDEX*)hnsw)->addPoint(tmp, label);
	  slock.rRelease();
      delete tmp;
    }
    else{
      slock.rLock();
      ((INDEX*)hnsw)->addPoint(&data[0], label);
      slock.rRelease();
    }
    return true;
  }

  template<class T>
  std::vector<std::pair<float, uint64_t>> FeatureDB<T>::query(const std::vector<T>& data,const unsigned k) {
    std::vector<std::pair<float, uint64_t>> result;
    CHECK_DIM(data.size(), Dim, result)
      if (normalize) {
        float *tmp = new float[Dim];
        normalization(&data[0], tmp, Dim);
        auto top_candidate = ((INDEX*)hnsw)->searchKnn(tmp, k);
		result.resize(top_candidate.size());
        unsigned count = top_candidate.size();
        for (int i = count - 1; i >= 0; --i) {
          result[i] = top_candidate.top();
          top_candidate.pop();
        }
        delete tmp;
      }
      else {
        auto top_candidate = ((INDEX*)hnsw)->searchKnn(&data[0], k);
        result.resize(top_candidate.size());
        unsigned count = top_candidate.size();
        for (int i = count - 1; i >= 0; --i) {
          result[i] = top_candidate.top();
          top_candidate.pop();
        }
      }
    return result;
  }

  template<class T>
  std::vector<std::pair<float, uint64_t>> FeatureDB<T>::queryAndInsert(const std::vector<T>& data,const uint64_t label,const unsigned k) {
    std::vector<std::pair<float, uint64_t>> result;
    CHECK_DIM(data.size(), Dim, result)

      if (normalize) {
        float* tmp = new float[Dim];
        normalization(&data[0], tmp, Dim);
        auto top_candidate = ((INDEX*)hnsw)->searchKnn(tmp, k);
        result.resize(top_candidate.size());
        unsigned count = top_candidate.size();
        for (int i = count - 1; i >= 0; --i) {
          result[i] = top_candidate.top();
          top_candidate.pop();
        }
        slock.rLock();
        ((INDEX*)hnsw)->addPoint(tmp, label);
        slock.rRelease();
        delete tmp;
      }
      else {
        auto top_candidate = ((INDEX*)hnsw)->searchKnn(&data[0], k);
        result.resize(top_candidate.size());
        unsigned count = top_candidate.size();
        for (int i = count - 1; i >= 0; --i) {
          result[i] = top_candidate.top();
          top_candidate.pop();
        }
        slock.rLock();
		((INDEX*)hnsw)->addPoint(&data[0], label);
        slock.rRelease();
      }
    return result;
  }

  template<class T>
  std::vector<std::pair<float, uint64_t>> FeatureDB<T>::queryAndInsert(const std::vector<T>& data,const uint64_t label,const float threshold,const unsigned k) {
    std::vector<std::pair<float, uint64_t>> result;
    CHECK_DIM(data.size(), Dim, result)

      if (normalize) {
        float* tmp = new float[Dim];
        normalization(&data[0], tmp, Dim);
        auto top_candidate = ((INDEX*)hnsw)->searchKnn(tmp, k);
        result.resize(top_candidate.size());
        unsigned count = top_candidate.size();
        for (int i = count - 1; i >= 0; --i) {
          result[i] = top_candidate.top();
          top_candidate.pop();
        }
        if (result[0].first > threshold){
          slock.rLock();
		  ((INDEX*)hnsw)->addPoint(tmp, label);
          slock.rRelease();
		}
        delete tmp;
      }
      else {
        auto top_candidate = ((INDEX*)hnsw)->searchKnn(&data[0], k);
        result.resize(top_candidate.size());
        unsigned count = top_candidate.size();
        for (int i = count - 1; i >= 0; --i) {
          result[i] = top_candidate.top();
          top_candidate.pop();
        }
        if (result[0].first > threshold){
          slock.rLock();
		  ((INDEX*)hnsw)->addPoint(&data[0], label);
          slock.rRelease();
		}
	  }
    return result;
  }

  template class FeatureDB<float>;
}
}


#include "FeatureDB.h"

namespace fdb {
  template<class T>
  FeatureDB<T>::FeatureDB(
    std::string space_name,
    const unsigned dim,
    const unsigned M,
    const unsigned max_elements,
    const unsigned query_ef,
    const unsigned construction_ef
    std::string dtype = "float")
    :Dim(dim) {
    normalize = false;
    initialize = false;
    hnswlib::SpaceInterface<T> *space;
    if (space_name == "ip") space = new hnswlib::InnerProductSpace(dim);
    else if (space_name == "l2") space = new hnswlib::L2Space(dim);
    else if (space_name == "cosine") {
      space = new hnswlib::InnerProductSpace(dim);
      normalize = true;
    }

    hnsw = new hnswlib::HierarchicalNSW<float>(space, max_elements, M, construction_ef);
    hnsw->ef_ = query_ef;
  }

  template<class T>
  FeatureDB<T>::~FeatureDB() {
    delete space;
    delete hnsw;
  }

  template<class T>
  FeatureDB<T>::void normalization(T* src, T* dst, unsigned dim) {
    T* tmp = src;
    T* sum = 0;
    for (int i = 0; i < dim; ++i) {
      sum += (*tmp)*(*tmp);
      ++tmp;
    }
    sum = sqrt(sum);
    for (int i = 0; i < dim; ++i)
      *(dst++) = *(src++) / sum;
  }

  template<class T>
  FeatureDB<T>::unsigned size() {
    return hnsw->cur_element_count;
  }

  template<class T>
  FeatureDB<T>::const std::vector<char>& dump() {
    std::vector<char> serial;
    hnsw->saveIndex(serial);
    return serial;
  }

  template<class T>
  FeatureDB<T>::bool load(const std::vector<char>& serial) {
    if (initialize) return false;
    initialize = true;
    hnsw->loadIndex(serial);
    return true;
  }

  #CHECK_DIM(obj, src) \
    if (obj != src) return false;

  template<class T>
  FeatureDB<T>::bool insert(std::vector<T> data, uint64_t label) {
    initialize = true;
    CHECK_DIM(data.size(), Dim)
      if (normalization) {
        float *tmp = new float[Dim];
        normalization(&data[0], tmp, Dim);
        hnsw->addPoint(tmp, label);
        delete tmp;
      }
      else
        hnsw->addPoint(&data[0], label);
    return true;
  }

  template<class T>
  FeatureDB<T>::std::vector<pair<float, uint64_t>>& query(std::vector<T> data, unsigned k) {
    std::vector<pair<float, uint64_t>> result;
    CHECK_DIM(data.size(), Dim)
      if (normalize) {
        float *tmp = new float[Dim];
        normalization(&data[0], tmp, Dim);
        auto top_candidate = hnsw->searchKnn(tmp, k);
        unsigned count = top_candidate.size();
        for (int i = count - 1; i >= 0; --i) {
          result[i] = top_candidate.top();
          top_candidate.pop();
        }
        delete tmp;
      }
      else {
        auto top_candidate = hnsw->searchKnn(tmp, k);
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
  FeatureDB<T>::std::vector<pair<float, uint64_t>> queryAndInsert(std::vector<T> data, uint64_t label, unsigned k, ) {
    std::vector<pair<float, uint64_t>> result;
    if (data.size() != Dim) return result;

    if (normalization) {
      float* tmp = new float[Dim];
      normalization(&data[0], tmp, Dim);
      auto top_candidate = hnsw->searchKnn(tmp, k);
      result.resize(top_candidate.size());
      unsigned count = top_candidate.size();
      for (int i = count - 1; i >= 0; --i) {
        result[i] = top_candidate.top();
        top_candidate.pop();
      }
      hnsw->addPoint(tmp, label);
      delete tmp;
    }
    else {
      auto top_candidate = hnsw->searchKnn(&data[0], k);
      std::vector<pair<float, uint64_t>> result;
      result.resize(top_candidate.size());
      unsigned count = top_candidate.size();
      for (int i = count - 1; i >= 0; --i) {
        result[i] = top_candidate.top();
        top_candidate.pop();
      }
      hnsw->addPoint(&data[0], label);
    }
    return result;
  }

  template<class T>
  FeatureDB<T>::std::vector<pair<uint64_t, float>> queryAndInsert(std::vector<T> data, uint64_t label, float threshold, unsigned k) {
    std::vector<pair<float, uint64_t>> result;
    if (data.size() != Dim) return result;

    if (normalization) {
      float* tmp = new float[Dim];
      normalization(&data[0], tmp, Dim);
      auto top_candidate = hnsw->searchKnn(tmp, k);
      result.resize(top_candidate.size());
      unsigned count = top_candidate.size();
      for (int i = count - 1; i >= 0; --i) {
        result[i] = top_candidate.top();
        top_candidate.pop();
      }
      if (result[0].first > threshold)
        hnsw->addPoint(tmp, label);
      delete tmp;
    }
    else {
      auto top_candidate = hnsw->searchKnn(&data[0], k);
      std::vector<pair<float, uint64_t>> result;
      result.resize(top_candidate.size());
      unsigned count = top_candidate.size();
      for (int i = count - 1; i >= 0; --i) {
        result[i] = top_candidate.top();
        top_candidate.pop();
      }
      if (result[0].first > threshold)
        hnsw->addPoint(&data[0], label);
    }
    return result;
  }

  template class FeatureDB<float>;
}

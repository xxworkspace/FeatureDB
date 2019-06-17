
#pragma once

#include <string>
#include <vector>
#include "hnswlib/hnswalg.h"

namespace fdb {
	template<class T = float>
	class FeatureDB {
	private:
		uint32_t Dim;
		bool normalize;
		bool initialize;
		hnswlib::HierarchicalNSW *hnsw;
		void normalization(T* src, T* dst, unsigned dim);
	public:
		FeatureDB(
			std::string space_name,
			const unsigned dim,
			const unsigned M,
			const unsigned max_elements,
			const unsigned query_ef,
			const unsigned construction_ef
			std::string dtype = "float");
		~FeatureDB();
		unsigned size();
		const std::vector<char>& dump();
		bool load(const std::vector<char>& serial);
		bool insert(std::vector<T> data, uint64_t label);
		std::vector<pair<float, uint64_t>>& query(std::vector<T> data, unsigned k);
		std::vector<pair<float, uint64_t>> queryAndInsert(std::vector<T> data, uint64_t label, unsigned k);
		std::vector<pair<uint64_t, float>> queryAndInsert(std::vector<T> data, uint64_t label, float threshold, unsigned k);
	};
}

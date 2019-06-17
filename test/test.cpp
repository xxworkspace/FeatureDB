
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "FeatureDB.h"

void fvecs(std::string filename,std::vector<float*> &data, int& dim) {
  std::ifstream ifs(filename, std::ios::binary);
  if (ifs.fail()){
    std::cout << "file open fail!" << std::endl;
    exit(0);
  }
  while (!ifs.eof()) {
    ifs.read((char*)&dim,sizeof(int));
	float* ftmp = (float*)malloc(dim*sizeof(float));
	ifs.read((char*)ftmp, sizeof(float)*dim);
	data.push_back(ftmp);
  }
  ifs.close();
}

int main(){
  int dim;
  std::vector<float*> data;
  fvecs("sift_query.fvecs",data,dim);
  
  fdb::FeatureDB<float> db("l2",dim,60,3600000,120,360);
  for(int i = 0 ; i < 1000 ; i++){
    std::vector<float> dt(data[i],data[i] + dim);
	db.insert(dt,i);
  }

  for(int i = 0 ; i < 100 ; i++){
    std::vector<float> qr(data[i],data[i] + dim);
    auto rs = db.query(qr,2);
    for(auto tmp : rs)
      std::cout<<tmp.first<<" "<<tmp.second<<tmp.second<<"  ||  ";
    std::cout<<std::endl;
  }
  
  auto serial = db.dump();
  return 0;
}


#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "FeatureDB.h"
#include <thread>
#include <mutex>
#include <omp.h>

void fvecs(std::string filename,std::vector<float*> &data, int& dim) {
  std::ifstream ifs(filename, std::ios::binary);
  if (ifs.fail()){
    std::cout << "file open fail!" << std::endl;
    exit(0);
  }
  while (true) {
    ifs.read((char*)&dim,sizeof(int));
    if(ifs.eof()) break;
    float* ftmp = (float*)malloc(dim*sizeof(float));
    ifs.read((char*)ftmp, sizeof(float)*dim);
    data.push_back(ftmp);
  }
  ifs.close();
}

int main(){
  int dim;
  std::vector<float*> data;
  //std::vector<float*> query;
  fvecs("sift_query.fvecs",data,dim);
  //fvecs("sift_query.fvecs",query,dim);
  
  bigo::ml::FeatureDB<float> db("cosine",dim,60,3600000,120,360);

#pragma omp parallel for
  for(int i = 0 ; i < data.size() ; ++i){
    std::vector<float> dt(data[i],data[i] + dim);
    db.insert(dt,i);
  }

#pragma omp parallel for
  for(int i = 0 ; i < data.size() ; ++i){
    std::vector<float> qr(data[i],data[i] + dim);
    auto rs = db.query(qr,2);
    for(auto tmp : rs)
      std::cout<<i<<" "<<tmp.first<<" "<<tmp.second<<"  ||  ";
    std::cout<<std::endl;
  }

  auto serial = db.dump();
  bigo::ml::FeatureDB<float> gdb("cosine",dim,60,3600000,120,360);
  gdb.load(serial);

  db.save("testdb");
  bigo::ml::FeatureDB<float> ldb("cosine",dim,60,3600000,120,360);
  ldb.restore("testdb");

#pragma omp parallel for
  for(int i = 0 ; i < data.size() ; ++i){
    std::vector<float> dt(data[i],data[i] + dim);
    ldb.insert(dt,i);
  }

/*
#pragma omp parallel for
  for(int i = 0 ; i < data.size() ; ++i){
    std::vector<float> qr(data[i],data[i] + dim);
    auto rs = ldb.query(qr,2);
    for(auto tmp : rs)
      std::cout<<i<<" "<<tmp.first<<" "<<tmp.second<<"  ||  ";
    std::cout<<std::endl;
  }
  */
/*
#pragma omp parallel for
  for(int i = 0 ; i < data.size() ; ++i){
    std::vector<float> qr(data[i],data[i] + dim);
    auto rs = gdb.query(qr,2);
    for(auto tmp : rs)
      std::cout<<i<<" "<<tmp.first<<" "<<tmp.second<<"  ||  ";
    std::cout<<std::endl;
  }
*/
  return 0;
}

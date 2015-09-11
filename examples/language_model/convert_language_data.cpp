// This script converts the MNIST dataset to a lmdb (default) or
// leveldb (--backend=leveldb) format used by caffe to load data.
// Usage:
//    convert_mnist_data [FLAGS] input_image_file input_label_file
//                        output_db_file
// The MNIST dataset could be downloaded at
//    http://yann.lecun.com/exdb/mnist/


#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#include <stdint.h>
#include <sys/stat.h>

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "caffe/proto/caffe.pb.h"
#include <vector>
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

const int vocab_size = 10003;
const int unknown_symbol = vocab_size -3;
const int start_symbol = vocab_size -2;
const int zero_symbol = vocab_size -1;
const int maximum_length = 30;
void convert_dataset(string phrase) {
  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  leveldb::WriteBatch* batch = NULL;
  string db_path = ".//examples//language_model//lm_" + phrase + "_leveldb";
  leveldb::Status status = leveldb::DB::Open(
      options, db_path, &db);

  batch = new leveldb::WriteBatch();
  const int kMaxKeyLength = 10;
  char key_cstr[kMaxKeyLength];
  string value;
  Datum datum;
  datum.set_channels(2*maximum_length);
  datum.set_height(1);
  datum.set_width(1);

  for (int i=0;i<2*maximum_length;i++)
    datum.add_float_data(0.0);


  string tmp = ".//data//language_model//"+phrase+"_indices.txt";
  std::ifstream infile(tmp);
  string s;
  int item_id = 0;
  while (getline(infile,s)){
    std::vector<float> dt,real_dt;
	  std::istringstream iss(s);
	  int num;
	  while (iss >> num){
		  if (num >= unknown_symbol)
			  num = unknown_symbol;
		  dt.push_back(num);
    }
    if (dt.size()<maximum_length){
      int l = maximum_length-dt.size();
      for (int i=0;i<l;i++)
        dt.push_back(zero_symbol);
    }
    real_dt.push_back(zero_symbol);
    for (int i=0;i<dt.size()-1;i++)
      real_dt.push_back(dt[i]);
    for (int i=0;i<dt.size();i++)
      real_dt.push_back(dt[i]);
    _snprintf(key_cstr, kMaxKeyLength, "%08d", item_id);
    string keystr(key_cstr);
    for (int i=0;i<2*maximum_length;i++)
      datum.set_float_data(i,real_dt[i]);
    datum.SerializeToString(&value);
    batch->Put(keystr, value);
    item_id++;
  }
  db->Write(leveldb::WriteOptions(), batch);
  delete batch;
}

int main(int argc, char** argv) {
  convert_dataset("train");
  convert_dataset("valid");
  convert_dataset("test");
  return 0;
}


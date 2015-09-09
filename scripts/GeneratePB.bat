if exist "../src/caffe/proto/caffe.pb.h" (
    echo caffe.pb.h remains the same as before
) else (
    echo caffe.pb.h is being generated
    "../3rdparty/bin/protoc" -I="../src/caffe/proto" --cpp_out="../src/caffe/proto" "../src/caffe/proto/caffe.proto"
)


if exist "../python/caffe/proto/caffe_pb2.py" (
    echo caffe_pb2.py remains the same as before
) else (
    echo caffe_pb2.py is being generated
    "../3rdparty/bin/protoc" -I="../src/caffe/proto" --python_out="../python/caffe/proto" "../src/caffe/proto/caffe.proto"
)
pause
::if exist "../src/caffe/proto/caffe_pretty_print.pb.h" (
::    echo caffe_pretty_print.pb.h remains the same as before
::) else (
 ::   echo caffe_pretty_print.pb.h is being generated
 ::   "../../3rdparty/bin/protoc" -I="../../src/caffe/proto" --cpp_out="../../src/caffe/proto" "../../src/caffe/proto/caffe_pretty_print.proto"
::)



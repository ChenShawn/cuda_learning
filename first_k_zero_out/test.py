import tensorflow as tf

ops = tf.load_op_library('./cuda_op_kernel.so')
print(ops)

with tf.Session() as sess:
    ans = sess.run(ops.add_one([1,2,3,4,5,6,7,8,9,10]))
    print(ans)


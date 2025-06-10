from model_test import test_relu, test_softmax, test_inference, load_test_acc

test_relu()
test_softmax()
test_inference()
print("✅ 準確率為：", load_test_acc())

# Prediction 1

Prediction: Vocabulary không tăng khi chuyển từ unigram sang bigram hoặc trigram.

Reason: Vocabulary vẫn là tập các từ duy nhất trong cùng corpus.

Confidence: High

# Prediction 2

Prediction: Số lượng unique n-gram tăng khi n tăng.

Reason: Bigram và trigram tạo nhiều tổ hợp từ hơn unigram.

Confidence: High

# Prediction 3

Prediction: Trigram dễ gặp zero probability nhất.

Reason: Context dài hơn làm dữ liệu thưa hơn và nhiều n-gram chưa từng xuất hiện.

Confidence: High

# Prediction 4

Prediction: Trigram có thể có training perplexity thấp nhất.

Reason: Context dài hơn giúp mô hình ghi nhớ training data tốt hơn.

Confidence: Medium

# Prediction 5

Prediction: Trigram không chắc tốt hơn bigram khi corpus nhỏ.

Reason: Corpus nhỏ làm trigram dễ bị sparsity và zero probability.

Confidence: High
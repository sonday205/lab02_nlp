# Next-word Prediction

| Context | Prediction | Probability | Actual |
|---|---|---:|---|
| machine learning | and | 0.000554 | and |
| in the | most | 0.006706 | server |
| one of | the | 0.103060 | the |
| of the | most | 0.006706 | most |
| to the | most | 0.006706 | next |

Actual word được xác định là từ xuất hiện thường xuyên nhất ngay sau context tương ứng trong validation set.

# Correct Predictions

Context: machine learning

Prediction: and

Expected: and

Probability: 0.000554

Reason: Bigram này đã xuất hiện đủ để mô hình học được quan hệ giữa các từ.

Context: one of

Prediction: the

Expected: the

Probability: 0.103060

Reason: Đây là một mẫu từ khá phổ biến trong corpus.

# Incorrect Predictions

Context: in the

Prediction: most

Expected: server

Probability: 0.006706

Reason: Mô hình chỉ dùng context ngắn nên không đủ thông tin để xác định từ tiếp theo chính xác.

Context: to the

Prediction: most

Expected: next

Probability: 0.006706

Reason: Có nhiều từ có thể xuất hiện sau context này, gây sparsity và làm dự đoán chưa chính xác.

# Sentence Ranking

Kết quả xếp hạng:

1. is useful for NLP
2. banana computer quickly
3. studies language models

Candidate đầu tiên có mean log probability cao nhất nên được xếp đầu.

Tuy nhiên, cách chấm điểm bằng bigram chỉ xét quan hệ giữa các từ gần nhau nên chưa đánh giá tốt ý nghĩa của toàn câu. Vì vậy một câu không tự nhiên như “banana computer quickly” vẫn có thể được xếp cao hơn một câu hợp lý hơn.

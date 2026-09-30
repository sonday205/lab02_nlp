# LAB 02 — N-gram Language Models

## Nội dung

- `ngram_lm.py`: cài đặt unigram, bigram, trigram; MLE/Laplace; xác suất câu, log probability, perplexity và dự đoán từ kế tiếp.
- `experiments.ipynb`: thống kê tối đa 10.000 tài liệu C4, chia train/validation/test theo tài liệu, khảo sát phân phối n-gram, đánh giá và ứng dụng.
- `results.csv`: số liệu đầu ra sau khi chạy notebook.
- `calculations.pdf`: bản scan phần tính toán viết tay.
- `error_analysis.md`: phân tích prediction và xếp hạng câu.
- `prediction.md`, `reflection.md`: cần được hoàn thiện trước khi nộp; nội dung hiện đang để trống.

## Dữ liệu

Notebook đọc `../c4-train.00000-of-01024-30K.json.gz`, shard C4 nằm ở thư mục cha. Chỉ dùng tối đa 10.000 tài liệu. Không cần tải dữ liệu bổ sung.

## Chạy

Từ thư mục `lab_02`:

```bash
python -m pip install pandas matplotlib nbformat jupyter
jupyter notebook experiments.ipynb
```

Chạy các cell từ trên xuống. Cell cuối sẽ ghi đè `results.csv` bằng số liệu của lần chạy hiện tại. Kết quả phụ thuộc shard, preprocessing, thư viện và môi trường chạy.

## Chính sách AI

AI hỗ trợ tạo phần code và khung thí nghiệm theo phạm vi cho phép trong PDF. Phần tính toán được lưu thành bản scan; prediction và reflection vẫn cần hoàn thiện. Error analysis hiện có trong `error_analysis.md`.

## AI assistance statement

- Tool: ChatGPT/Codex.
- Purpose: hỗ trợ tạo implementation Python và khung notebook cho các phần AI được phép theo PDF.
- What was generated: `ngram_lm.py` và code thí nghiệm trong `experiments.ipynb`.
- What was modified: code được điều chỉnh để dùng bộ đếm chung khi đánh giá MLE/Laplace và chia train/validation/test theo tài liệu.
- How the result was verified: notebook chạy hết trên shard dữ liệu hiện có; `results.csv` được tạo bởi cell cuối. Phần prediction và reflection do sinh viên tự hoàn thiện trước khi nộp.

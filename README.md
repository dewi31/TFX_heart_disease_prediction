# Submission 1: Heart Disease Prediction
Nama: Dewi Wahidatul Karimah

Username dicoding: dewi_karimah

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Heart Disease Prediction Dataset](https://www.kaggle.com/datasets/mfarhaannazirkhan/heart-dataset) |
| Masalah | Penyakit jantung adalah kondisi ketika bagian jantung yang meliputi pembuluh darah jantung, selaput jantung, katup jantung, dan otot jantung mengalami gangguan. Penyakit jantung bisa disebabkan oleh berbagai hal, seperti sumbatan pada pembuluh darah jantung, peradangan, infeksi, atau kelainan bawaan. |
| Solusi machine learning | Ciri ciri orang yang mempunyai resiko besar terkena penyakit jantung bisa diketahui sejak dini sehingga bisa segera dilakukan proses penyembuhan. Oleh karena itu, model neural network yang dibuat cocok untuk pendeteksi dini orang yang memiliki resiko penyakit jantung. Dengan model ini juga diharapkan dapat membantu proses deteksi dengan cepat dan biaya lebih sedikit.|
| Metode pengolahan | Data Heart Disease Prediction memiliki 14 fitur dengan rincian satu fitur float dan sisanya integer. Data kemudian dibagi ke dalam 80% data latih dan 20% data uji. Data tersebut selanjutnya dilakukan normalisasi (kecuali fitur "target") agar rentang data sama menjadi antara 0 sampai 1. |
| Arsitektur model |Arsitektur model neural network terdiri dari tiga layer, satu input layer, satu hidden layer, dan satu output layer. Selain itu, model juga menggunakan fungsi aktivasi relu dan sigmoid sebagai output. Untuk dibagian model fit fungsi loss menggunakan binary_crossentropy dengan optimizer Adam dan metrik BinaryAccuray. Dilakukan juga hyperparameter tuning pada komponen tuner dengan metode random search guna menemukan jumlah unit hidden layer dan learning rate yang menghasilkan akurasi terbaik.  Hasil parameter terbaik tersebut diteruskan ke bagian komponen trainer.|
| Metrik evaluasi | Metrik evaluasi yang digunakan yaitu ExampleCount, AUC, FalsePositives, TruePositives, FalseNegatives, TrueNegatives, Loss, dan BinaryAccuracy |
| Performa model | Dengan ExampleCount sebesar 367 menghasilkan AUC sebesar 0.873,  BinaryAccuracy sebesar 0.79, FalseNegatives sebesar 55, FalsePositives sebesar 22, Loss sebesar 0.657, TrueNegatives sebesar 151, dan TruePositives sebesar 139. Model yang dibuat cukup baik dalam melakukan prediksi tetapi performa model masih bisa ditingkatkan.|

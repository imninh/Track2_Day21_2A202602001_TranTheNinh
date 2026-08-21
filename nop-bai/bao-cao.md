# Bao Cao Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Ho va ten | Tran The Ninh |
| MSSV | 2A202602001 |
| Lop / Khoa | K4 |
| Repo GitHub | https://github.com/imninh/Track2_Day21_2A202602001_TranTheNinh |
| Ngay nop | 2026-08-21 |

---

## 1. Bo Sieu Tham So Da Chon va Ly Do

| Lan chay | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |

**Bo sieu tham so da chon:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Ly do:** Bo nay cho f1_score lop duong cao nhat (0.7149), dat nguong 0.65 cua lab. Quan trong: lan chay co accuracy cao nhat (0.8780, Run 1) khong trung voi lan co f1 cao nhat (Run 3), chung to accuracy khong phan anh dung chat luong tren du lieu mat can bang. Giua n_estimators va learning_rate co danh doi ro ret: giam learning_rate xuong 0.05 (Run 2) lam f1 tut xuong 0.6051 (duoi nguong) tru khi bu bang n_estimators lon hon.

---

## 2. Vi Sao Nguong Chat Luong Dat Tren F1 Chu Khong Phai Accuracy

Tap Adult co ty le lop thu nhap cao chi 24.8%. He qua, mot mo hinh "luon tra loi thu nhap thap" dat accuracy toi 0.752 nhay bat duoc dung 0 truong hop thu nhap cao (f1 = 0). Accuracy cao o day gay hieu nham vi bi lop da so keo len. f1_score cua lop duong (pos_label=1) do dong thoi precision va recall cua nhom thuc su quan trong - nguoi thu nhap cao - dieu accuracy khong do duoc. Ta khong dung average="weighted" hay "macro" vi gia tri do bi lop da so (75%) keo len cao, che lap viec mo hinh bo lo lop thieu so.

---

## 3. Kho Khan Gap Phai va Cach Giai Quyet

| Kho khan | Nguyen nhan | Cach giai quyet |
|---|---|---|
| Test pytest loi MLflow file store | Khong co tracking URI mac dinh khi chay test | Dat mac dinh sqlite:///mlflow.db trong train() neu thieu env |
| Mot cau hinh (50/0.05/2) rot nguong 0.65 | learning_rate qua nho, boosting chua hoi tu | Quet nguong quyet dinh (Bonus 2) tim nguong 0.30 day f1 len 0.7368 |
| Doi cloud GCP sang AWS | Lab yeu cau chon 1 provider | Doi google-cloud-storage thanh boto3, serve.py dung S3 |

---

## 4. So Sanh Buoc 2 va Buoc 3 (bat buoc, 2 - 3 cau)

| | f1_score | accuracy |
|---|---|---|
| Buoc 2 (chi `train_batch1`) | 0.7149 | 0.8740 |
| Buoc 3 (them `train_batch2`) | 0.7354 | 0.8820 |

**Nhan xet:** Them `train_batch2` (cung phan phoi) giup f1 tang nhe 0.020 len 0.7354 do mo hinh hoc on dinh hon tren nhieu du lieu hon, khong phai vi thong tin moi.

---

## 5. Phan Bonus Da Thuc Hien (neu co)

- [x] Bonus 1 - Tracking MLflow tu xa voi DagsHub: tich hop qua env MLFLOW_TRACKING_URI trong cicd.yml.
- [x] Bonus 2 - Dieu chinh nguong quyet dinh: quet 0.1-0.9, nguong toi uu 0.30 (f1 0.7368) so voi 0.5 (0.7149).
- [x] Bonus 3 - Bao cao precision / recall tu dong: outputs/detail.txt + upload artifact.
- [x] Bonus 4 - Hoan tra ve phien ban truoc: so sanh f1 moi/cu tu s3://.../artifacts/current/report.json.
- [x] Bonus 5 - Canh bao lech lac du lieu: kiem tra ty le lop duong so voi 24.8%, canh bao neu lech >5pp.

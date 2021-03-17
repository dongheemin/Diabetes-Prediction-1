# data_science

본 레포는 데이터 과학 상지대학교 석사 민동희, 홍경찬의 데이터 과학 연구를 보관하는 저장소입니다.

주요 연구
<HR />

1. Diabetes Prediction Model
- 학습 데이터 : 2018 국민건강영양조사
- 데이터 분할 : 학습데이터(70%), 검증(10%), 실험(20%)
- Variable Selection Model : Rogistic Regression
- Feature : 신장, 체중, 허리둘레, BMI, 체중변화여부, 체중조절여부, 연간음주빈도, 음주시작연령, 흡연여부, 흡연시작연령, 현재흡연여부
- Output : 당뇨병 의사진단 여부
- 테스트 검증 : MLP With z-score-norm (87%), RNN(87%), GRU(88%), LSTM(57%)
- 샘플 데이터
- patient_1 = np.array([[156  , 68.1, 81.2, 27.98,    2,    4,    0,  0,    0,    0,    0]]) # dg = 1 (비공개)
- patient_2 = np.array([[157.1, 64.2, 86.8, 26.01,    1,    1, 	  0,  0,	  3,  	0,    0]]) # dg = 1 (국민건강)
- patient_3 = np.array([[177.9, 74.7, 80.4, 23.60,	  1,	  1,	  2, 16,    2,   16,    3]]) # dg = 0 (국민건강)

- 샘플 데이터1 검증 : 전부 10% 이하
- 샘플 데이터2 검증 : MLP (12%), RNN (30%), GRU (88%), LSTM (87%)
- 샘플 데이터3 검증 : MLP(100%), RNN(100%), GRU(100%), LSTM(100%)


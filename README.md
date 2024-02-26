# K-ium 의료인공지능 최우수상
![KakaoTalk_20240226_221151059](https://github.com/KSI000321/Kium-/assets/122200920/5c5864a1-51a0-42a0-ade3-2fda806b5ec2)

# run.py 실행 방법

## 파일 경로 수정
run.py에는 각 이미지, csv파일, 모델들의 주소들을 포함하는 경로가 있는데, 각각 '#번호'로 표시를 해 두었습니다(#1~ #9). 이 주소만 수정하고 실행하시면 output.csv을 지정된 주소에 출력할 수 있습니다. 제출파일의 절대 경로인 file_path 변수만 수정 하면 되겠으나, 오류가 난다면 '#번호'로 표시된 곳에 각각 절대경로들을 입력해주세요!

### #1, #2, #3 (csv file, image paths)
![image](https://github.com/KSI000321/git_practice/assets/122200920/6679781d-00e2-47b3-83bc-eff1ef88637d)
  * #1 : 키움화이팅_제출파일 절대 경로
  * #2 : test.csv 주소
  * #3 : test image (I/V) 주소들

### #4, #5, #6, #7 (trained model paths)
![image](https://github.com/KSI000321/git_practice/assets/122200920/3389c09c-f555-4672-97d8-c6230b5f3992)

  * #4 : SwinNet_ant_multilabel.pt - Timm
  * #5 : ResNet_pos_multilabel.pt - torchvision (사진에선 SwinNet이라 잘못 표기)
  * #6 : MedNet_ant_binary.pt - https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet10/blob/main/resnet_10.pth
  * #7 : MedNet_pos_binary.pt

### #8 (output.csv save path)
![image](https://github.com/KSI000321/git_practice/assets/122200920/35adf274-fc08-42df-bed1-9e9d59dc8fa2)
  * #8 : output.csv 저장 위치

## 디렉토리들의 위치
![image](https://github.com/KSI000321/git_practice/assets/122200920/3e8cc9a3-4d11-4a34-8f77-3d37c064e344)
* model 파일의 resnet_18.pth는 추론에 사용되는 모델이 fine-tuning에 사용되는 가중치임 => train.py에서만 사용 

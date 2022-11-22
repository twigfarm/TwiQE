# TwiQE

### Predict
#### Step1. Call model
```python
model = TwiQE('ko', 'en', './your/path/to/model.pkl')
```
or
```python
model = TwiQE(source_language_code='ko', target_language_code='en', score_model_path='./your/path/to/model.pkl')
```

The following language pairs are supported:  
ko-en  
en-ko

#### Step2. Do predict
```python
sentences = [
    ["구체적이어야 합니다.", 
     "Our guests always enjoy taking them home as a souvenir."],
    ["완료되었으므로 지금 시청할 수 있습니다.", 
     "Thousands of Korean dramas await you!"],
    ["이러한 방법을 통해 우리 학교를 깨끗이 하여 쾌적한 환경에서 생활할 수 있을 것입니다.",
     "In these ways, we can live in a more pleasurable environment."],
    ["아이아스는 '장님'이 돼 소 떼와 양 떼를 마구잡이로 죽인다.", 
     "Ias becomes a 'jang' and kills herds and sheep recklessly."],
    ["그런 걸 할 수 있다는 말을 저희한테 안 하시고!", 
     "I can't believe you never told us you could produce something like that!"],
    ["진한 홍차로 주문하겠습니다.", 
     "I'd like a strong one, please."],
    ["향후 계속된 예후 관찰이 필요하다고 생각됩니다.", 
     "It was thought that it was necessary for constant watch out. "],
    ["안녕하세요, 저는 이제 반 50살이 됐습니다.", 
     "Hello, I'm 25 years old this year."],
    ["네가 기대할 만하다고 해서 지금 엄청나게 기대하고 있어.", 
     "I'm can't wait now because you said it's worth looking forward to."],
    ["같은 줄로 하프를 연주하지 마십시오.", 
     "Don't harp on the same string."]
]
predictions = model.predict(sentences)
```

### Train
#### Step1. Initialize model
```python
model = TwiQE('ko', 'en')
```
or
```python
model = TwiQE(source_language_code='ko', target_language_code='en')
```

#### Step2. Train model
```
sentences = [
     ["대신 병원에서 여태까지 만들어진 것을 보고 싶다고 하세요.",
     "Instead, show us the progress of what has been made so far in the customized program of the hospital."],
     ["물론 귀사의 선박에 전혀 문제는 없습니다.",
     "Of course, there is no problem with your ship at all."],
     ["요즘 같은 겨울철에는 전기 담요가 인기를 끌고 있습니다.",
     "Electric blankets are gaining popularity in winter like these days."],
     ["따라서 최고의 재료를 사용해야 해요.",
     "Thanks a lot for your help, and I'll be looking forward to your message."],
     ["혼합 현실 및 확장 현실과 관련된 사실적인 모델을 만듭니다.",
     "No sir, that is handled by a different department."],
     ["당신네 나라 사람들은 시네마에서 그것을 보기 위해 돈을 지불하지만, 현지인들은 1달러도 안 되는 돈으로 그들의 방에서 편안하게 그것을 즐긴다.",
     "While people in your country pay to see it in Cinemas, locals enjoy it at the comfort of their rooms, with less than $1"],
     ["네 맞아요 제가 다리가 좀 불편해서 평지 아파트면 좋겠습니다 1층도 상관없어요",
     "Yes that's right"],
     ["만약 아직 제품들을 배송하지 않으셨다면 최대한 빨리 배송을 해 주시기 바랍니다.",
     "If you haven't shipped the products yet, please ship them as soon as possible."],
    ]
labels = [71, 92, 83, 34, 22, 89, 21, 96]

model.train(sentences, labels)
```
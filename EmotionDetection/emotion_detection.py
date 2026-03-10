import requests
import json

def emotion_detector(text_to_analyse):
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    payload = { "raw_document": { "text": text_to_analyse } }
    response = requests.post(url, json=payload, headers=headers)

    response = json.loads(response.text)

    # Sample:
    # '{"emotionPredictions":[{"emotion":{"anger":0.0132405795, "disgust":0.0020517302, "fear":0.009090992, "joy":0.9699522, "sadness":0.054984167}, 
    #                          "target":"", 
    #                          "emotionMentions":[{"span":{"begin":0, "end":26, "text":"I love this new technology"}, "emotion":{"anger":0.0132405795, "disgust":0.0020517302, "fear":0.009090992, "joy":0.9699522, "sadness":0.054984167}}]}
    #                        ], 
    #           "producerId":{"name":"Ensemble Aggregated Emotion Workflow", "version":"0.0.1"}
    #  }'
    scores = response["emotionPredictions"][0]["emotion"]

    dominant = (None, 0)
    for k,v in scores.items():
        if v > dominant[1]:
            dominant = (k, v)
    
    scores['dominant_emotion'] = dominant[0]


    return json.dumps(scores, indent=4)
 
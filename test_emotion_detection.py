from EmotionDetection.emotion_detection import emotion_detector

TEST_VECTORS = [
    ("I am glad this happened", "joy"),
    ("I am really mad about this", "anger"),
    ("I feel disgusted just hearing about this", "disgust"),
    ("I am so sad about this", "sadness"),
    ("I am really afraid that this will happen", "fear"),
]

def test_emotion_detection():
    for test in TEST_VECTORS:
        scores = emotion_detector(test[0])
        assert scores['dominant_emotion'] == test[1]

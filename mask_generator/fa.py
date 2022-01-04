import face_alignment
from skimage import io

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

input = io.imread('./data/fake/1.jpg')
preds = fa.get_landmarks(input)
preds_list=preds[0].astype(int).tolist()
print(preds_list)
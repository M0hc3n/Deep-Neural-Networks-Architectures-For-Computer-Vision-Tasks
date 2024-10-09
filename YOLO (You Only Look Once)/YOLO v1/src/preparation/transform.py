

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for transform in self.transforms:
            img, bboxes = transform(img), bboxes
        return img, bboxes
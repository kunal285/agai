from dentistry.quality import assess_image
class ImageQualityAgent:
    name="image_quality"
    def run(self, image, config=None): return {"agent":self.name, **assess_image(image,config)}

import t2v_metrics
clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl') # our recommended scoring model

### For a single (image, text) pair
image = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/baselines/results/z_image/LongText-Bench/result_longtext_en_72.png" # an image path in string format
text = "A visually appealing social media interface shown on a smartphone screen. At the very top, the app's name is displayed clearly as \"FitLife\", styled in bold, modern typography, colored in energizing bright orange. Underneath this title, a smaller, engaging subtitle reads, \"Empower yourself, every step counts\". At the main section of the screen, there's a featured user post with prominent textual content reading, \"Completed my first 10K run today!\" Beneath this highlighted achievement, a supportive and motivational description appears in smaller but easily readable text, \"Feeling accomplished and motivated to keep pushing myself further. Huge thanks to the FitLife community for endless inspiration!\" Just below this user-generated content, there are clearly visible social interaction buttons labeled \"Like\", \"Comment\", and \"Share\" in neat, neatly aligned blocks, each paired with subtle, simplistic icons. Towards the bottom of the screen, a neatly organized navigation bar with concise labels like \"Home\", \"Explore\", \"Progress\", and \"Profile\" appears clearly legible and easily accessible, adding functionality and encouraging seamless exploration of the content. The interface utilizes warm colors and sleek fonts, fostering an inviting and uplifting online community environment."
score = clip_flant5_score(images=[image], texts=[text])
print(score)
### Alternatively, if you want to calculate the pairwise similarity scores 
### between M images and N texts, run the following to return a M x N score tensor.
# images = ["images/0.png", "images/1.png"]
# texts = ["someone talks on the phone angrily while another person sits happily",
#          "someone talks on the phone happily while another person sits angrily"]
# scores = clip_flant5_score(images=images, texts=texts) # scores[i][j] is the score between image i and text j
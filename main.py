import os
import argparse
import utils
import torch
from google_images_download import google_images_download
import sys
sys.path.append('ControlNet')
import canny as cy

parser = argparse.ArgumentParser(description='Description of your script.')

parser.add_argument('--main_object', type=str, default='a bird', help='Text description of the main object in the picture you want')
parser.add_argument('--prompt', type=str, default='a bird with black foot and brown wings on the beach', help='Text description of the picture you want')
parser.add_argument('--dataset', type=str, default='web', help='Input web for using web images, input the file path for using a local database')
parser.add_argument('--prompt_gen', type=str, default='template', help='using llama2 or template')
parser.add_argument('--output_path', type=str, default='images/')
parser.add_argument('--employ_filter', action='store_true', help='A boolean flag')
parser.add_argument('--re_num', type=int, default=10, help='number of images generated per image input to ControlNet')
parser.add_argument('--target_img', type=str, default='')
args = parser.parse_args()
print(args)

# get retrieval image
if args.dataset == 'web':
	print('web')
	response = google_images_download.googleimagesdownload()

	arguments = {
		"keywords": args.prompt,
		"limit": 5,
		"print_urls": True,
		"chromedriver": "path2chromedriver",
		"output_directory": args.output_path
	}
	paths, errors = response.download(arguments)
	r_file = args.output_path + '/' + os.listdir(args.output_path + args.prompt)[0]
else:
	r_file = utils.t2t_cmp(args.dataset, args.prompt)
torch.cuda.empty_cache()

# get synthetic image
if args.prompt_gen == 'template':
	re_prompt = utils.template_gen(args.main_object, args.prompt)
	sd_img = utils.prompt2img(re_prompt, args.output_path)
elif args.prompt_gen == 'llama2':
	re_prompt = utils.llama2_gen(args.main_object, args.prompt)
	sd_img = utils.prompt2img(re_prompt, args.output_path)
torch.cuda.empty_cache()

# ControlNet re-generate image
file_list = [r_file]
file_list += [os.path.join(sd_img, i) for i in os.listdir(sd_img)]
print(file_list)
if args.employ_filter:
	img = utils.t2i_cmp(args.prompt, file_list)
	cy.base_canny([img], args.main_object, args.prompt, args.re_num)
else:
	cy.base_canny(file_list, args.main_object, args.prompt, args.re_num)

# eval
img_path = args.output_path + "result"
file_list = [os.path.join(img_path, i) for i in os.listdir(img_path)]
best, sim = utils.t2i_eval(args.prompt, file_list)
print("-"*20)
print("best similarity with input text: %.2f" %sim)
print("image path: %s" % best)
if args.target_img != '':
	best, sim = utils.i2i_eval(args.target_img, file_list)
	print("best similarity with target image: %.2f" %sim)
	print("image path: %s" % best)

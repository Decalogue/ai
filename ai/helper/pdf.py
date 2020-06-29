# -*- coding: utf-8 -*-
""" ai.helper.pdf """

import pdfbox
from pathlib import Path


p = pdfbox.PDFBox()
end_tags = [
	'Introduction', 'introduction', 'Introductoin', 'INTRODUCTION',
	'Motivation', 'Background', 'motivation', 'background'
	]


def extract_abstract(filepath, start_tag='Abstract', end_tags=end_tags):
	""" Extract abstract from a PDF-formatted scientific and technological article
		从 PDF 格式的科技论文中抽取摘要
	"""
	text = p.extract_text(filepath, 'pdf.txt')
	res = []
	flag = False
	with open('pdf.txt', 'r', encoding='utf-8') as f:
		for l in f.readlines():
			if start_tag in l:
				flag = True
			elif flag:
				l = l.strip()
				if l == '':
					continue
				for t in end_tags:
					if t in l or t.lower() in l:
						if res:
							res[-1] = res[-1].rstrip()
						return ''.join(res)
				if l.endswith('-'):
					l = l[:-1]
				else:
					l += ' '
				res.append(l)
	if res:
		res[-1] = res[-1].rstrip()
	return ''.join(res)
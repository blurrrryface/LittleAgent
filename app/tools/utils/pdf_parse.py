import base64
import os
import re
import uuid

import fitz  # PyMuPDF
import shapely.geometry as sg
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from shapely.validation import explain_validity

from app.tools.models.select_llms import get_llm_by_name
from app.tools.utils.oss_upload import OssUploader


class PdfParse:
    def __init__(self, llm_name: str = "gpt4o"):
        self.llm = get_llm_by_name(llm_name)
        self.uploader = OssUploader()

    def _is_near(self, rect1, rect2, distance=20):
        """
        check if two rectangles are near each other if the distance between them is less than the target
        """
        # 检测两个矩形的距离是否小于distance
        return rect1.buffer(1).distance(rect2.buffer(1)) < distance

    def _is_horizontal_near(self, rect1, rect2, distance=100):
        """
        check if two rectangles are near horizontally if one of them is a horizontal line
        """
        result = False
        # rect1和rect2中有一个是水平线
        if abs(rect1.bounds[3] - rect1.bounds[1]) < 0.1 or abs(rect2.bounds[3] - rect2.bounds[1]) < 0.1:
            if abs(rect1.bounds[0] - rect2.bounds[0]) < 0.1 and abs(rect1.bounds[2] - rect2.bounds[2]) < 0.1:
                result = abs(rect1.bounds[3] - rect1.bounds[3]) < distance
        # print(rect1.bounds, rect2.bounds, result)
        return result

    def _union_rects(self, rect1, rect2):
        """
        union two rectangles
        """
        # 合并两个矩形
        return sg.box(*(rect1.union(rect2).bounds))

    def _merge_rects(self, rect_list, distance=20, horizontal_distance=None):
        """
        merge rectangles in the list if the distance between them is less than the target
        :param rect_list: list of rectangles
        :param distance: distance threshold
        :param horizontal_distance: horizontal distance threshold when one of the rectangles is a horizontal line
        """
        # 合并矩形列表
        merged = True
        while merged:
            merged = False
            new_rect_list = []
            while rect_list:
                rect = rect_list.pop(0)
                for other_rect in rect_list:
                    if self._is_near(rect, other_rect, distance) or (
                            horizontal_distance and self._is_horizontal_near(rect, other_rect, horizontal_distance)):
                        rect = self._union_rects(rect, other_rect)
                        rect_list.remove(other_rect)
                        merged = True
                new_rect_list.append(rect)
            rect_list = new_rect_list
        return rect_list

    def _adsorb_rects_to_rects(self, source_rects, target_rects, distance=10):
        """
        吸附一个集合到另外一个集合
        :param source_rects: 源矩形集合
        :param target_rects: 目标矩形集合
        """
        new_source_rects = []
        for text_area_rect in source_rects:
            adsorbed = False
            for index, rect in enumerate(target_rects):
                if self._is_near(text_area_rect, rect, distance):
                    rect = self._union_rects(text_area_rect, rect)
                    target_rects[index] = rect
                    adsorbed = True
                    break
            if not adsorbed:
                new_source_rects.append(text_area_rect)
        return target_rects, new_source_rects

    def _parse_drawings(self, page):
        """
        parse drawings in the page and merge adjacent rectangles
        """
        """
        解析页面中的绘图元素，合并相邻的矩形
        """
        drawings = page.get_drawings()

        # for drawing in drawings:
        #     print(drawing)

        rect_list = [drawing['rect'] for drawing in drawings]
        # 转成 shapely 的 矩形对象
        rect_list = [sg.box(rect[0], rect[1], rect[2], rect[3]) for rect in rect_list]

        merged_rects = self._merge_rects(rect_list, distance=10, horizontal_distance=100)

        # 并删除无效的矩形
        merged_rects = [rect for rect in merged_rects if explain_validity(rect) == 'Valid Geometry']

        # 提取所有的字符串区域矩形框
        text_area_rects = []
        for text_area in page.get_text('dict')['blocks']:
            rect = text_area['bbox']
            text_area_rects.append(sg.box(rect[0], rect[1], rect[2], rect[3]))

        # 吸附文字到矩形区域
        merged_rects, text_area_rects = self._adsorb_rects_to_rects(text_area_rects, merged_rects, distance=5)

        # 二次合并
        merged_rects = self._merge_rects(merged_rects, distance=10)

        # 过滤掉高度 或者 宽度 不足 10 的矩形
        merged_rects = [rect for rect in merged_rects if
                        rect.bounds[2] - rect.bounds[0] > 10 and rect.bounds[3] - rect.bounds[1] > 10]

        # 将Polygon对象抽取bounds属性
        merged_rects = [rect.bounds for rect in merged_rects]

        return merged_rects

    def _parse_images(self, page):
        """
        解析页面中的图片元素
        """
        images = page.get_image_info()
        return [image['bbox'] for image in images]

    def _parse_tables(self, page):
        """
        解析页面中的表格元素
        """
        tables = page.find_tables(
            snap_tolerance=20,  # 调整容差以捕捉更多的表格线条
        )
        return [table.bbox for table in tables]

    def _parse_rects(self, page):
        """
        parse rectangles in the page
        :param page: page object
        """
        """
        解析页面中的矩形元素
        """
        return self._parse_images(page) + self._parse_drawings(page)

    def _parse_pdf_to_images(self, pdf_path, output_dir='../../../data/temp'):
        """
        parse pdf to images and save to output_dir
        :param pdf_path: pdf file path
        :param output_dir: output directory
        :return: image_infos [(page_image, rect_images)]
        """
        import os
        # 打开PDF文件
        pdf_document = fitz.open(pdf_path)

        image_infos = []
        for page_index, page in enumerate(pdf_document):
            print(f'parse page: {page_index}')
            # 保存页面为图片
            page_image = page.get_pixmap(matrix=fitz.Matrix(3, 3))

            rect_images = []
            # 解析页面中的矩形
            # if page_index != 5:
            #     continue
            rects = self._parse_rects(page)
            # rects = _parse_tables(page)
            for index, rect in enumerate(rects):
                # print(page_index, index, rect)
                fitz_rect = fitz.Rect(rect)
                pix = page.get_pixmap(clip=fitz_rect, matrix=fitz.Matrix(4, 4))
                name = f'{page_index}_{index}.png'
                pix.save(os.path.join(output_dir, name))
                # 存储最简相对路径
                rect_images.append(name)

                # 在页面上绘制红色矩形(膨胀一个像素)
                # page.draw_rect(fitz_rect, color=(1, 0, 0), width=1)
                big_fitz_rect = fitz.Rect(fitz_rect.x0 - 1, fitz_rect.y0 - 1, fitz_rect.x1 + 1, fitz_rect.y1 + 1)
                page.draw_rect(big_fitz_rect, color=(1, 0, 0), width=1)

                # 在矩形内的左上角写上矩形的索引name，添加一些偏移量
                text_x = fitz_rect.x0 + 2  # 偏移量为2个单位
                text_y = fitz_rect.y0 + 10  # 偏移量为10个单位
                text_rect = fitz.Rect(text_x, text_y - 9, text_x + 80, text_y + 2)  # 创建一个白色背景的矩形

                # 绘制白色背景矩形
                page.draw_rect(text_rect, color=(1, 1, 1), fill=(1, 1, 1))

                # 插入带有白色背景的文字
                page.insert_text((text_x, text_y), name, fontsize=10, color=(1, 0, 0))

            # 重新生成带有矩形的页面图像
            page_image_with_rects = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            # 整页使用绝对路径
            page_image = os.path.join(output_dir, f'{page_index}.png')
            page_image_with_rects.save(page_image)

            image_infos.append((page_image, rect_images))

        # 关闭PDF文件
        pdf_document.close()
        return image_infos

    # def encode_image(self, image_path):
    #     with open(image_path, "rb") as image_file:
    #         return base64.b64encode(image_file.read()).decode("utf-8")

    def multimodal_gpt_chain(self, system_prompt, user_prompt, llm, image_url):
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(system_prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{image_url}"},
                    },
                ]
            )]
        )

        chain = prompt | llm
        return chain.invoke({}).content

    def replace_specific_image(self,text, image_name, url):
        # 构建精确匹配指定图片名称的正则表达式
        pattern = re.escape(image_name)  # 对image_name进行转义，确保特殊字符被正确处理
        pattern = f'!\[{pattern}\]\({pattern}\)'

        def replacer(match):
            # 直接返回替换后的完整Markdown图片链接
            return f'![{image_name}]({url})'

        # 使用re.sub进行替换，并返回结果
        return re.sub(pattern, replacer, text)


    def _gpt_parse_images(self, image_infos, llm, output_dir='../../../data/temp'):
        """
        parse images to markdown content
        :param image_infos: [(page_image, rect_images)]
        :param output_dir: output directory
        :param api_key: OpenAI API Key
        :param base_url: OpenAI Base URL
        :param model: OpenAI Vison LLM Model
        :return: markdown content
        """
        import os

        prompt = """
        使用markdown语法，输出图片的全部内容。
        内容不要包含在```markdown ```中、段落公式使用 $$ $$ 的形式、行内公式使用 $ $ 的形式、忽略掉长直线、忽略掉页码。
        不要解释，直接输出内容。
        """

        rect_prompt = """
        图片中用红色框和名称(%s)标注出了一些区域。
        如果区域是表格或者图片，使用 ![]() 的形式插入到输出内容中，否则直接输出文字内容。
        """

        role = '你是一个PDF文档解析器，使用markdown和latex语法输出图片的内容。'

        contents = []
        for index, (page_image, rect_images) in enumerate(image_infos):
            page_image_uuid = str(uuid.uuid4())
            page_image_filename = os.path.basename(page_image)
            page_image_unique = f"{os.path.splitext(page_image_filename)[0]}_{page_image_uuid}{os.path.splitext(page_image_filename)[1]}"
            page_img_url = self.uploader.upload_image(page_image,page_image_unique)

            print(f'gpt parse page: {index}')

            local_prompt = prompt
            if rect_images:
                local_prompt += rect_prompt % ', '.join(rect_images)


            content = self.multimodal_gpt_chain(role, local_prompt, llm, page_img_url)
            if rect_images:
                for rect_image in rect_images:
                    rect_images_uuid = str(uuid.uuid4())
                    rect_images_filename = os.path.basename(rect_image)
                    rect_images_unique = f"{os.path.splitext(rect_images_filename)[0]}_{rect_images_uuid}{os.path.splitext(rect_images_filename)[1]}"
                    rect_img_url = self.uploader.upload_image(os.path.join(output_dir, rect_image),rect_images_unique)
                    content = self.replace_specific_image(content,rect_image, rect_img_url)
            contents.append(content)

        # 输出结果
        output_path = os.path.join(output_dir, 'output.md')
        with open(output_path, 'w') as f:
            f.write('\n\n'.join(contents))

        return '\n\n'.join(contents)

    def parse_pdf(self, pdf_path, output_dir='../../../data/temp'):
        """
        解析PDF文件到markdown文件
        :param pdf_path: pdf文件路径
        :param output_dir: 输出目录。存储所有的图片和markdown文件
        :param api_key: OpenAI API Key（可选）。如果未提供，则使用OPENAI_API_KEY环境变量。
        :param base_url: OpenAI Base URL。 （可选）。如果未提供，则使用OPENAI_BASE_URL环境变量。
        :param model: OpenAI Vison LLM Model，默认为'gpt-4o'。您还可以使用qwen-vl-max
        :param verbose: 详细模式，默认为False
        :return: (content, all_rect_images), markdown内容，带有![](path/to/image.png) 和 所有矩形图像（图像、表格、图表等）路径列表。
        """

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        image_infos = self._parse_pdf_to_images(pdf_path, output_dir=output_dir)
        content = self._gpt_parse_images(image_infos, llm=self.llm, output_dir=output_dir)

        # 删除每页的图片 & 保留所有的矩形图片
        all_rect_images = []
        for page_image, rect_images in image_infos:
            if os.path.exists(page_image):
                os.remove(page_image)
            if os.path.exists(page_image):
                os.remove(rect_images)
            # all_rect_images.extend(rect_images)

        return content  # , all_rect_images

    # def parse_pdf_and_save(self, pdf_path, output_dir='../../../data/temp'):
    #     """
    #     正则解析出剩余的图片， 并对图片进行上传 将上传后得图片链接替换现在得图片
    #
    #     """
    #     content, all_rect_images = self.parse_pdf(pdf_path, output_dir=output_dir)
    #     import re
    #     content = re.sub(r'!\[.*?\]\(.*?\)', lambda m: m.group(0).replace('![]', '!'), content)
    #
    #
    #     with open(output_path, 'w') as f:
    #         f.write(content)
    #     return content, all_rect_images

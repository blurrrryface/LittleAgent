import oss2
import os

from dotenv import load_dotenv

load_dotenv()

class OssUploader:
    def __init__(self):
        """
        初始化OSS客户端
        :param endpoint: OSS服务的访问域名
        :param access_key_id: 阿里云账号AccessKey ID
        :param access_key_secret: 阿里云账号AccessKey Secret
        :param bucket_name: 您的存储空间名称（Bucket Name）
        """
        endpoint = os.getenv("OSS_ENDPOINT")
        access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
        access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")
        bucket_name = os.getenv("OSS_BUCKET_NAME")

        self.auth = oss2.Auth(access_key_id, access_key_secret)
        self.bucket = oss2.Bucket(self.auth, endpoint, bucket_name)

    def upload_image(self, local_file_path, remote_file_name=None):
        """
        上传图片到OSS
        :param local_file_path: 本地图片文件路径
        :param remote_file_name: 图片上传到OSS后的文件名，默认与本地文件名相同
        :return: 图片在OSS上的URL，如果上传失败则返回None
        """
        if not remote_file_name:
            remote_file_name = os.path.basename(local_file_path)
        try:
            with open(local_file_path, 'rb') as fileobj:
                self.bucket.put_object(remote_file_name, fileobj)
            # 构建图片URL
            image_url = f"https://{self.bucket.bucket_name}.{self.bucket.endpoint.replace('http://','')}/{remote_file_name}"
            return image_url
        except FileNotFoundError:
            print(f"文件不存在: {local_file_path}")
            return None
        except Exception as e:
            print(f"图片上传失败: {e}")
            return None

# 使用示例
if __name__ == "__main__":
    uploader = OssUploader()

    # 确保使用正确的文件路径，这里演示了使用绝对路径的检查
    local_image_path = os.path.abspath(os.path.join(os.getcwd(), "../../../data/prompt设计.md"))
    # if not os.path.exists(local_image_path):
    #     print(f"文件不存在: {local_image_path}")
    #     exit(1)  # 如果文件不存在，则终止程序

    remote_url = uploader.upload_image(local_image_path)
    if remote_url:
        print(f"图片上传成功，访问URL: {remote_url}")
    else:
        print("图片上传失败")

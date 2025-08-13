from .kumikolib import Kumiko
import tempfile, cv2, os

def get_panels_from_array(img_rgb, rtl=True):

    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    path = tmp.name
    tmp.close()  

    cv2.imwrite(path, img_rgb)

    k = Kumiko({'rtl': rtl})
    k.parse_image(path)
    infos = k.get_infos()

    os.unlink(path)

    return infos[0]['panels']
import tempfile
import subprocess
import math
import cv2

from ..utils import Context

alignment_to_justification = {'left': 'TEXT-JUSTIFY-LEFT', 'right': 'TEXT-JUSTIFY-RIGHT', 'center': 'TEXT-JUSTIFY-CENTER'}

text_init_template = '''
            ( text{n} ( car ( gimp-text-layer-new image "{text}" "Noto Sans" {text_size} 0 ) ) )
'''

# text rotation was buggy so I left it out
#    ( gimp-item-transform-rotate text{n} {angle} TRUE 0 0 )

text_template = '''
    ( gimp-image-add-layer image text{n} 0 )
    ( gimp-text-layer-set-color text{n} (list {color}) )
    ( gimp-item-set-name text{n} "{name}" )
    ( gimp-layer-set-offsets text{n} {position} )
    ( gimp-text-layer-resize text{n} {size} )
    ( gimp-text-layer-set-justification text{n} {justify} )
'''

save_tempaltes = {
    'xcf': '( gimp-xcf-save RUN-NONINTERACTIVE image inpainting "{out_file}" "{out_file}" )',
    'psd': '( file-psd-save RUN-NONINTERACTIVE image inpainting "{out_file}" "{out_file}" 0 0 )',
    'pdf': '( file-pdf-save RUN-NONINTERACTIVE image inpainting "{out_file}" "{out_file}" TRUE TRUE TRUE )',
}

script_template = '''
( let* (
            ( image ( car ( gimp-file-load RUN-NONINTERACTIVE "{input_file}" "{input_file}" ) ) )
            ( layer-list (gimp-image-get-layers image))
            ( background_layer (car layer-list))
            ( inpainting ( car ( gimp-file-load-layer RUN-NONINTERACTIVE image "{mask_file}" ) ) )
            {text_init}
        )
    ( gimp-image-add-layer image inpainting 0 )
    ( gimp-item-set-name inpainting "mask" )
    ( gimp-item-set-name background_layer "original image" )
    {text}
    {save}
    ( gimp-quit 0 )                        
)'''

def gimp_render(out_file, ctx: Context):
    input_file = tempfile.NamedTemporaryFile(suffix='.png')
    mask_file = tempfile.NamedTemporaryFile(suffix='.png')

    ctx.input.save(input_file.name)
    cv2.imwrite(mask_file.name, ctx.gimp_mask)

    extension = out_file.split('.')[-1]

    text_init = ''.join([text_init_template.format(
        text=text_region.translation.replace('"', '\\"'), text_size=text_region.font_size, n=n
        ) for n, text_region in enumerate(ctx.text_regions)])

    text = ''.join([text_template.format(
        n=n, color=' '.join([str(value) for value in text_region.fg_colors]),
        name=' '.join(text_region.text), position=str(text_region.xywh[0])+' '+str(text_region.xywh[1]),
        size=str(text_region.xywh[2])+' '+str(text_region.xywh[3]),
        justify=alignment_to_justification[text_region.alignment],
        angle=math.radians(text_region.angle),
        ) for n, text_region in enumerate(ctx.text_regions)])

    full_script = script_template.format(
        input_file=input_file.name, mask_file=mask_file.name, text_init=text_init, text=text,
        extension=extension, save=save_tempaltes[extension].format(out_file=out_file)
    )

    subprocess.run(['gimp', '-i', '-b', full_script])
    input_file.close()
    mask_file.close()
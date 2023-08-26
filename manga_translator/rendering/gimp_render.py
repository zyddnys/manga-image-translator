import tempfile
import subprocess
import math
import cv2
import platform
import glob
import os

from ..utils import Context

DEFAULT_FONT = 'Sans-serif'

alignment_to_justification = {'left': 'TEXT-JUSTIFY-LEFT', 'right': 'TEXT-JUSTIFY-RIGHT', 'center': 'TEXT-JUSTIFY-CENTER'}
direction_to_base_direction = {'h': 'TEXT-DIRECTION-LTR', 'v': 'TEXT-DIRECTION-TTB-LTR-UPRIGHT', 'hr': 'TEXT-DIRECTION-RTL', 'vr': 'TEXT-DIRECTION-TTB-RTL-UPRIGHT'}

text_init_template = '( text{n} ( car ( gimp-text-layer-new image "{text}" "{default_font}" {text_size} 0 ) ) )'
font_template = '( gimp-text-layer-set-font text{n} "{font}" )'
angle_template = '( gimp-item-transform-rotate text{n} {angle} TRUE 0 0 )'

text_template = '''
    ( gimp-image-add-layer image text{n} 0 )
    ( gimp-text-layer-set-color text{n} (list {color}) )
    ( gimp-item-set-name text{n} "{name}" )
    ( gimp-layer-set-offsets text{n} {position} )
    ( gimp-text-layer-resize text{n} {size} )
    ( gimp-text-layer-set-language text{n} "{language}" )
    ( gimp-text-layer-set-letter-spacing text{n} {letter_spacing} )
    ( gimp-text-layer-set-line-spacing text{n} {line_spacing} )
    ( gimp-text-layer-set-base-direction text{n} {base_direction} )
    ( gimp-text-layer-set-justification text{n} {justify} )
    {font}
    {angle}
'''

save_tempaltes = {
    'xcf': '( gimp-xcf-save RUN-NONINTERACTIVE image background_layer "{out_file}" "{out_file}" )',
    'psd': '( file-psd-save RUN-NONINTERACTIVE image background_layer "{out_file}" "{out_file}" 0 0 )',
    'pdf': '( file-pdf-save RUN-NONINTERACTIVE image background_layer "{out_file}" "{out_file}" TRUE TRUE TRUE )',
}

create_mask = '( inpainting ( car ( gimp-file-load-layer RUN-NONINTERACTIVE image "{mask_file}" ) ) )'
rename_mask = '( gimp-image-add-layer image inpainting 0 ) ( gimp-item-set-name inpainting "mask" )'

script_template = '''
( let* (
            ( image ( car ( gimp-file-load RUN-NONINTERACTIVE "{input_file}" "{input_file}" ) ) )
            ( layer-list (gimp-image-get-layers image))
            ( background_layer (car layer-list))
            {create_mask}
            {text_init}
        )
    {rename_mask}
    ( gimp-item-set-name background_layer "original image" )
    {text}
    {save}
    ( gimp-quit 0 )                        
)'''

def gimp_render(out_file, ctx: Context):
    input_file = os.path.join(tempfile.gettempdir(), '.gimp_input.png')
    mask_file = os.path.join(tempfile.gettempdir(), '.gimp_mask.png')

    extension = out_file.split('.')[-1]

    ctx.input.save(input_file)
    if ctx.gimp_mask is not None:
        cv2.imwrite(mask_file, ctx.gimp_mask)
    else:
        ctx.text_regions = []

    text_init = ''.join([text_init_template.format(
            n=n,
            text=text_region.translation.replace('"', '\\"'),
            text_size=text_region.font_size,
            default_font=DEFAULT_FONT+(' Bold' if text_region.bold else '')+(' Italic' if text_region.italic else '')
        ) for n, text_region in enumerate(ctx.text_regions)])

    text = ''.join([text_template.format(
            n=n,
            color=' '.join([str(value) for value in text_region.fg_colors]),
            name=' '.join(text_region.text),
            position=str(text_region.xywh[0])+' '+str(text_region.xywh[1]),
            size=str(text_region.xywh[2])+' '+str(text_region.xywh[3]),
            justify=alignment_to_justification[text_region.alignment],
            font=font_template.format(n=n, font=text_region.font_family) if text_region.font_family != '' else '',
            angle=angle_template.format(n=n, angle=math.radians(text_region.angle)) if abs(text_region.angle) > 10 else '',
            language=text_region.target_lang,
            line_spacing=text_region.line_spacing,
            letter_spacing=text_region.letter_spacing,
            base_direction=direction_to_base_direction[text_region.direction],
        ) for n, text_region in enumerate(ctx.text_regions)])

    full_script = script_template.format(
        input_file=input_file,
        text_init=text_init,
        text=text,
        extension=extension,
        save=save_tempaltes[extension].format(out_file=out_file.replace('\\', '\\\\')),
        create_mask=(create_mask.format(mask_file=mask_file) if ctx.gimp_mask is not None else ''),
        rename_mask=(rename_mask if ctx.gimp_mask is not None else ''),
    )

    executable = 'gimp'
    if platform.system() == 'Windows':
        gimp_dir = os.getenv('LOCALAPPDATA')+'\\Programs\\GIMP 2\\bin\\'
        executables = glob.glob(gimp_dir+'gimp-console-2.*.exe')
        if len(executables) == 0:
            print('error: gimp not found in directory:', gimp_dir)
            return
        executable = executables[0]
    
    subprocess.run([executable, '-i', '-b', full_script])

    os.unlink(input_file)
    os.unlink(mask_file)
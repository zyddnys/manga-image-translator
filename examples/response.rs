#[derive(Debug)]
struct Color {
    fg: [u8; 3],
    bg: [u8; 3],
}

#[derive(Debug)]
struct Translation {
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    is_bulleted_list: bool,
    angle: u32,
    prob: f32,
    text_color: Color,
    text: HashMap<String, String>,
    background: Vec<u8>,
}

#[derive(Debug)]
struct TranslationResponse {
    translations: Vec<Translation>,
}

impl TranslationResponse {
    fn from_bytes(bytes: &[u8]) -> Self {
        let mut offset = 0;
        let v = (0..read_u32(bytes, &mut offset))
            .map(|_| Translation::from_bytes(bytes, &mut offset))
            .collect::<Vec<_>>();
        Self { translations: v }
    }
}

fn read_u32(bytes: &[u8], offset: &mut usize) -> u32 {
    let value = u32::from_le_bytes(bytes[*offset..*offset + 4].try_into().unwrap());
    *offset += 4;
    value
}

impl Translation {
    fn from_bytes(bytes: &[u8], offset: &mut usize) -> Self {
        let read_f32 = |bytes: &[u8], offset: &mut usize| -> f32 {
            let value = f32::from_le_bytes(bytes[*offset..*offset + 4].try_into().unwrap());
            *offset += 4;
            value
        };

        let read_u8 = |bytes: &[u8], offset: &mut usize| -> u8 {
            let value = bytes[*offset];
            *offset += 1;
            value
        };

        let read_bool = |bytes: &[u8], offset: &mut usize| -> bool {
            let value = bytes[*offset] != 0;
            *offset += 1;
            value
        };

        let read_chunk = |bytes: &[u8], offset: &mut usize| -> Vec<u8> {
            let size = read_u32(bytes, offset);
            let value = &bytes[*offset..*offset + size as usize];
            *offset += size as usize;
            value.to_vec()
        };
        let read_str = |bytes: &[u8], offset: &mut usize| -> String {
            String::from_utf8(read_chunk(bytes, offset)).expect("Invalid UTF-8")
        };
        let read_map = |bytes: &[u8], offset: &mut usize| -> HashMap<String, String> {
            (0..read_u32(bytes, offset))
                .into_iter()
                .map(|_| (read_str(bytes, offset), read_str(bytes, offset)))
                .collect::<HashMap<String, String>>()
        };
        Self {
            min_x: read_u32(bytes, offset),
            min_y: read_u32(bytes, offset),
            max_x: read_u32(bytes, offset),
            max_y: read_u32(bytes, offset),
            is_bulleted_list: read_bool(bytes, offset),
            angle: read_u32(bytes, offset),
            prob: read_f32(bytes, offset),
            text_color: Color {
                fg: [
                    read_u8(bytes, offset),
                    read_u8(bytes, offset),
                    read_u8(bytes, offset),
                ],
                bg: [
                    read_u8(bytes, offset),
                    read_u8(bytes, offset),
                    read_u8(bytes, offset),
                ],
            },
            text: read_map(bytes, offset),
            background: read_chunk(bytes, offset),
        }
    }
}

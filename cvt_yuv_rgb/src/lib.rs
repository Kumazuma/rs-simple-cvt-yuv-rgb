#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
pub mod color_primaries{
    pub const BT601:ColorPrimary=ColorPrimary{kr:0.299,kg:0.587, kb:0.114}; 
    pub const BT709:ColorPrimary=ColorPrimary{kr:0.2126,kg:0.7152, kb:0.0722}; 
    pub const BT2020:ColorPrimary=ColorPrimary{kr:0.2627,kg:0.678, kb:0.0593}; 
    pub struct ColorPrimary{
        kr:f32,
        kg:f32,
        kb:f32
    }
    impl ColorPrimary{
        pub fn kr(&self)->f32{self.kr}
        pub fn kg(&self)->f32{self.kg}
        pub fn kb(&self)->f32{self.kb}
    }
    
}
#[derive(Clone,Debug)]
pub struct YUV{
    y:f32,
    u:f32,
    v:f32
}
impl YUV{
    pub fn new(y:f32, u:f32, v:f32)->Self{
        Self{
            y:y,
            u:u,
            v:v
        }
    }
}
impl Copy for YUV {}
#[derive(Clone,Debug)]
pub struct RGB{
    r:f32,
    g:f32,
    b:f32
}
impl RGB{
    pub fn new(r:f32, g:f32, b:f32)->Self{
        let r = if r > 1.0{1.0}
        else if r < 0.0{0.0}
        else{r};
        let g = if g > 1.0{1.0}
        else if g < 0.0{0.0}
        else{g};
        let b = if b > 1.0{1.0}
        else if b < 0.0{0.0}
        else{b};
        Self{
            r: r,
            g:g,
            b:b
        }
    }
}
impl Copy for RGB {}
struct Plane<'a>{
    data:&'a[u8],
    stride:u32
}
impl<'a> Plane<'a>{
    fn row<T:Sized>(&self, at:usize)->&[T]{
        let start = at * self.stride as usize;
        let end =  (at + 1) * self.stride as usize;
        unsafe{
            let row = &self.data[start..end];
            let len = row.len();
            let len = len / (std::mem::size_of::<T>()/std::mem::size_of::<u8>());
            &std::mem::transmute::<_,&[T]>(row)[0..len]
        }
    }
}
pub struct YUVImage<'a>
{
    plane_y:Plane<'a>,
    plane_u:Plane<'a>,
    plane_v:Plane<'a>,
    chroma_shift_x:u8,
    chroma_shift_y:u8,
    width:u32,
    height:u32,
    depth:u32
}
impl<'a> YUVImage<'a>{
    pub fn iter(&self)->YUVImageIter<'_, 'a>{
        YUVImageIter::new(self)
    }
} 
pub struct YUVImageIter<'b, 'a:'b>{
    reader:&'b YUVImage<'a>,
    index:usize
}
impl<'b, 'a:'b> YUVImageIter<'b, 'a>{
    fn new(reader:&'b YUVImage<'a>)->Self{
        Self{
            reader:reader,
            index:0
        }
    }
}
impl<'b, 'a:'b> Iterator for YUVImageIter<'b, 'a>{
    type Item = YUV;
    fn next(&mut self)->Option<Self::Item>{
        if self.index >= (self.reader.height * self.reader.width) as usize{
            None
        }
        else{
            let row = self.index / self.reader.width as usize;
            let col = self.index % self.reader.width as usize;
            self.index += 1;

            let depth = ((1 << self.reader.depth) - 1) as f32;
            
            let shift_x = self.reader.chroma_shift_x;
            let shift_y = self.reader.chroma_shift_y;
            let (y, u, v) = if self.reader.depth > 8{
                (self.reader.plane_y.row::<u16>(row)[col] as f32,
                self.reader.plane_u.row::<u16>(row>>shift_y)[col>>shift_x] as f32,
                self.reader.plane_v.row::<u16>(row>>shift_y)[col>>shift_x] as f32)
            }
            else{
                (
                self.reader.plane_y.row::<u8>(row)[col] as f32,
                self.reader.plane_u.row::<u8>(row>>shift_y)[col>>shift_x] as f32,
                self.reader.plane_v.row::<u8>(row>>shift_y)[col>>shift_x] as f32)
            };
            let cy = y / depth ;
            let cu = u / depth - 0.5;
            let cv = v / depth - 0.5;
            Some(YUV::new(cy,cu,cv))
        }
    }
}
pub struct YUVImageBuilder<'a>{
    plane_y:Option<Plane<'a>>,
    plane_u:Option<Plane<'a>>,
    plane_v:Option<Plane<'a>>,
    chroma_shift_x:Option<u8>,
    chroma_shift_y:Option<u8>,
    width:Option<u32>,
    height:Option<u32>,
    depth:Option<u32>,
}
impl<'a> YUVImageBuilder<'a>{
    pub fn new()->Self{
        Self{
            plane_y:None,
            plane_u:None,
            plane_v:None,
            chroma_shift_x:None,
            chroma_shift_y:None,
            width:None,
            height:None,
            depth:None
        }
    }
    pub fn plane_y(mut self, data:&'a[u8], stride:u32)->Self{
        self.plane_y =Some(Plane{
            data:data,
            stride:stride
        });
        return self;
    }
    pub fn plane_u(mut self, data:&'a[u8], stride:u32)->Self{
        self.plane_u =Some(Plane{
            data:data,
            stride:stride
        });
        return self;
    }
    pub fn plane_v(mut self, data:&'a[u8], stride:u32)->Self{
        self.plane_v =Some(Plane{
            data:data,
            stride:stride
        });
        return self;
    }
    pub fn chroma_shift_x(mut self, val:u8)->Self{
        self.chroma_shift_x = Some(val);
        return self;
    }
    pub fn chroma_shift_y(mut self, val:u8)->Self{
        self.chroma_shift_y = Some(val);
        return self;
    }
    pub fn width(mut self, val:u32)->Self{
        self.width = Some(val);
        return self;
    }
    pub fn height(mut self, val:u32)->Self{
        self.height = Some(val);
        return self;
    }
    pub fn depth(mut self, val:u32)->Self{
        self.depth = Some(val);
        return self;
    }
    pub fn build(self)->YUVImage<'a>{
        YUVImage{
            plane_y:self.plane_y.expect("missing plane_y"),
            plane_u:self.plane_u.expect("missing plane_u"),
            plane_v:self.plane_v.expect("missing plane_v"),
            chroma_shift_x:self.chroma_shift_x.expect("missing chroma_shift_x"),
            chroma_shift_y:self.chroma_shift_y.expect("missing chroma_shift_y"),
            width:self.width.expect("missing width"),
            height:self.height.expect("missing height"),
            depth:self.depth.expect("missing depth")
        }
    }
}

pub struct YUV2RGB<Reader:Iterator<Item=YUV>>{
    yuv_reader:Reader,
    color_primary:color_primaries::ColorPrimary
}
impl<Reader:Iterator<Item=YUV>> YUV2RGB<Reader>{
    pub fn new(reader:Reader, color_primary:color_primaries::ColorPrimary)->Self{
        Self{
            yuv_reader:reader,
            color_primary:color_primary
        }
    }
}
impl<Reader:Iterator<Item=YUV>> Iterator for YUV2RGB<Reader>{
    type Item = RGB;
    fn next(&mut self)->Option<Self::Item>{
        match self.yuv_reader.next(){
            None=>None,
            Some(yuv)=>{
                let kr = self.color_primary.kr();
                let kg = self.color_primary.kg();
                let kb = self.color_primary.kb();
                let R = yuv.y + (2f32 * (1f32 - kr)) * yuv.v;
				let B = yuv.y + (2f32 * (1f32 - kb)) * yuv.u;
				let G = yuv.y - ((2f32 * ((kr * (1f32 - kr) * yuv.v) + (kb * (1f32 - kb) * yuv.u))) / kg);
				Some(RGB::new(R,G,B))
            }
        }
    }
}
pub struct RGB2YUV<Reader:Iterator<Item=RGB>>{
    rgb_reader:Reader,
    color_primary:&'static color_primaries::ColorPrimary
}
impl<Reader:Iterator<Item=RGB>> RGB2YUV<Reader>{
    pub fn new(reader:Reader, color_primary:&'static color_primaries::ColorPrimary)->Self{
        Self{
            rgb_reader:reader,
            color_primary:color_primary
        }
    }
}
impl<Reader:Iterator<Item=RGB>> Iterator for RGB2YUV<Reader>{
    type Item = YUV;
    fn next(&mut self)->Option<Self::Item>{
        match self.rgb_reader.next(){
            None=>None,
            Some(rgb)=>{
                let kr = self.color_primary.kr();
                let kg = self.color_primary.kg();
                let kb = self.color_primary.kb();
                let y = kr * rgb.r + kg * rgb.g + kb * rgb.b;
				let u = 0.5 * ((rgb.b - y) / (1.0 - kb));
				let v = 0.5 * ((rgb.r - y) / (1.0 - kr));
				Some(YUV::new(y,u,v))
            }
        }
    }
}


pub struct RGB24Writer<Reader>
where Reader:Iterator<Item=RGB>{
    rgb:Reader,
    depth:u8
}
impl<Reader>  RGB24Writer<Reader>
where Reader:Iterator<Item=RGB>{
    pub fn new(_2rgb:Reader)->Self{
        Self{
            rgb:_2rgb,
            depth:8u8
        }
    }
}
impl<Reader> Iterator for RGB24Writer<Reader>
where Reader:Iterator<Item=RGB>{
    type Item = [u8;3];
    fn next(&mut self)->Option<Self::Item>{
        let max_bits=(1u32 << self.depth) - 1;
        let max = max_bits as f32;
        let rgb = self.rgb.next()?;
        let r = (rgb.r * max) as u8;
        let g = (rgb.g * max) as u8;
        let b = (rgb.b * max) as u8;
        return Some([r,g,b]);
    }
}
impl<Reader> std::io::Read for RGB24Writer<Reader>
where Reader:Iterator<Item=RGB>{
    fn read(&mut self, buf:&mut [u8])->Result<usize, std::io::Error>{
        let mut offset = 0usize;
        let max_bits=(1u32 << self.depth) - 1;
        let max = max_bits as f32;
        while buf.len() - offset >= 3{
            match self.rgb.next(){
                None if offset == 0 =>return Err(std::io::Error::from(std::io::ErrorKind::WriteZero)),
                None if offset != 0 =>return Ok(offset),
                Some(rgb)=>{
                    let r =std::cmp::min((rgb.r * max) as u8,  max_bits as u8);
                    let g = std::cmp::min((rgb.g * max) as u8, max_bits as u8);
                    let b = std::cmp::min((rgb.b * max) as u8, max_bits as u8);
                    buf[offset + 0] = r;
                    buf[offset + 1] = g;
                    buf[offset + 2] = b;
                    offset += 3;
                }
                _=>unreachable!()
            }    
        }
        return Ok(offset);
    }
    
}
pub struct RGB32Reader<Reader>
where Reader:Iterator<Item=RGB>{
    rgb:Reader,
    depth:u8
}
impl<Reader>  RGB32Reader<Reader>
where Reader:Iterator<Item=RGB>{
    pub fn new(_2rgb:Reader, depth:u8)->Self{
        Self{
            rgb:_2rgb,
            depth:depth
        }
    }
}
impl<Reader> std::io::Read for RGB32Reader<Reader>
where Reader:Iterator<Item=RGB>{
    fn read(&mut self, buf:&mut [u8])->Result<usize, std::io::Error>{
        let mut offset = 0usize;
        let max_bits=(1 << self.depth) - 1;
        let max = max_bits as f32;
        while buf.len() - offset >= 4{
            
            match self.rgb.next(){
                None if offset == 0 =>return Err(std::io::Error::from(std::io::ErrorKind::WriteZero)),
                None if offset != 0 =>return Ok(offset),
                Some(rgb)=>{
                    let r =std::cmp::min((rgb.r * max) as u32,  max_bits);
                    let g = std::cmp::min((rgb.g * max) as u32, max_bits);
                    let b = std::cmp::min((rgb.b * max) as u32, max_bits);
                    let _4byte:u32 =
                    r << (32 - self.depth * 1) |
                    g << (32 - self.depth * 2) |
                    b << (32 - self.depth * 3);
                    let _4byte = unsafe{std::mem::transmute::<_, [u8;4]>(_4byte)};
                    buf[offset + 0] = _4byte[0];
                    buf[offset + 1] = _4byte[1];
                    buf[offset + 2] = _4byte[2];
                    buf[offset + 3] = _4byte[3];
                    offset += 4;
                }
                _=>unreachable!()
            }    
        }
        return Ok(offset);
    }
    fn read_to_end(&mut self, buf: &mut Vec<u8>)->std::io::Result<usize>{
        let mut write_byte_count = 0usize;
        let max_bits=(1 << self.depth) - 1;
        let max = max_bits as f32;
        
        while let Some(rgb) = self.rgb.next(){
            let r = std::cmp::min((rgb.r * max) as u32,  max_bits);
            let g = std::cmp::min((rgb.g * max) as u32, max_bits);
            let b = std::cmp::min((rgb.b * max) as u32, max_bits);
            let _4byte:u32 =
            r << (32 - self.depth * 1) |
            g << (32 - self.depth * 2) |
            b << (32 - self.depth * 3);
            let _4byte = unsafe{std::mem::transmute::<_, [u8;4]>(_4byte)};
            buf.push(_4byte[0]);
            buf.push(_4byte[1]);
            buf.push(_4byte[2]);
            buf.push(_4byte[3]);
            write_byte_count += 4; 
        }
        return Ok(write_byte_count);
    }
}
pub struct RGB32ImageBuilder<'a>{
    data:Option<&'a [u8]>,
    width:Option<usize>,
    height:Option<usize>,
    stride:Option<usize>,
    depth:Option<u8>
}
impl<'a> RGB32ImageBuilder<'a>{
    pub fn new()->Self{
        Self{
            data:None,
            width:None,
            height:None,
            stride:None,
            depth:None
        }
    }
    pub fn data(mut self, data:&'a [u8])->Self{
        self.data = Some(data);
        self
    }
    pub fn width(mut self, val:usize)->Self{
        self.width = Some(val);
        self
    }
    pub fn height(mut self, val:usize)->Self{
        self.height = Some(val);
        self
    }
    pub fn stride(mut self, val:usize)->Self{
        self.stride = Some(val);
        self
    }
    pub fn depth(mut self, val:u8)->Self{
        self.depth = Some(val);
        self
    }
    pub fn build(self)->RGB32Image<'a>{
        RGB32Image{
            data:self.data.expect("data missing"),
            width:self.width.expect("width missing"),
            height:self.height.expect("height missing"),
            stride:self.stride.expect("stride missing"),
            depth:self.depth.expect("depth missing")
        }
    }
}
pub struct RGB32Image<'a>{
    data:&'a [u8],
    width:usize,
    height:usize,
    stride:usize,
    depth:u8
}
impl<'a> RGB32Image<'a>{
    pub fn iter(&self)->RGB32ImageIter<'_, 'a>{
        RGB32ImageIter::new(self)
    }
}
pub struct RGB32ImageIter<'a:'b, 'b>{
    image:&'b RGB32Image<'a>,
    index:usize
}
impl<'a:'b, 'b> RGB32ImageIter<'a,'b>{
    fn new(image:&'b RGB32Image<'a> )->Self{
        Self{
            image:image,
            index:0
        }
    }
}
impl<'a:'b, 'b> Iterator for RGB32ImageIter<'a,'b>{
    type Item = RGB;
    fn next(&mut self)->Option<Self::Item>{
        if self.index >= (self.image.height * self.image.width) as usize{
            return None;
        }
        let max_bits=(1 << self.image.depth) - 1;
        let max = max_bits as f32;
        let now_index = self.index;

        self.index += 1;
        let height_at = now_index / self.image.width ;
        let width_at = now_index % self.image.width ;
        let stride = self.image.stride;
        let row = &self.image.data[height_at * stride .. (height_at + 1) * stride];
        let _4bytes = &row[width_at * 4 .. (width_at + 1)*4];
        let _4bytes =unsafe{ std::mem::transmute::<_, u32>([_4bytes[0],_4bytes[1],_4bytes[2],_4bytes[3]])};
        let depth = self.image.depth;
        let r = (_4bytes >> (32 - depth*1)) & max_bits;
        let g = (_4bytes >> (32 - depth*2)) & max_bits;
        let b = (_4bytes >> (32 - depth*3)) & max_bits;
        let r = r as f32/ max;
        let g = g as f32/ max;
        let b = b as f32/ max;
        let rgb = RGB::new(r,g,b);
        return Some(rgb);
    }
}
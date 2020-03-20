#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
use std::u16;
use std::u32;
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
enum Plane<'a>{
    RefMut{data:&'a mut [u8], stride:u32},
    Ref{data:&'a [u8], stride:u32}
}
impl<'a> Plane<'a>{
    fn row<T:Sized>(&self, at:usize)->&[T]{
        let (data, stride) = match self{
            Plane::Ref{data,stride}=>(*data, *stride),
            Plane::RefMut{data, stride}=>(*data as &[u8],*stride)
        };
        let start = at * stride as usize;
        let end =  (at + 1) * stride as usize;
        unsafe{
            let row = &data[start..end];
            let len = row.len();
            let len = len / (std::mem::size_of::<T>()/std::mem::size_of::<u8>());
            &std::mem::transmute::<_,&[T]>(row)[0..len]
        }
    }
    fn row_mut<T>(&mut self, at:usize)->Option<&mut [T]>{
        let (data, stride) = match  self{
            Plane::Ref{..}=>return None,
            Plane::RefMut{data, stride}=>(data as &mut [u8],*stride)
        };
        let start = at * stride as usize;
        let end =  (at + 1) * stride as usize;
        
        unsafe{
            let end = if data.len() < end{
                data.len()
            }
            else{
                end
            };
            let row = &mut data[start..end];
            let len = row.len();
            let len = len / (std::mem::size_of::<T>()/std::mem::size_of::<u8>());
            
            Some(&mut std::mem::transmute::<_,&mut [T]>(row)[0..len])
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
pub trait YuvCompatible{
    fn y_row(&self, at:usize)->&[u8];
    fn y_row_mut(&mut self, at:usize)->&mut [u8];
    fn u_row(&self, at:usize)->&[u8];
    fn u_row_mut(&mut self, at:usize)->&mut [u8];
    fn v_row(&self, at:usize)->&[u8];
    fn v_row_mut(&mut self, at:usize)->&mut [u8];
    fn chroma_shift_x(&self)->u8;
    fn chroma_shift_y(&self)->u8;
    fn depth(&self)->u8;
    fn width(&self)->usize;
    fn height(&self)->usize;
    fn write_y(&mut self, row:usize, col:usize, value:u8){
        let row_y = self.y_row_mut(row);
        row_y[col] = value;
    }
    fn write_u(&mut self, row:usize, col:usize, value:u8){
        let row = row >> self.chroma_shift_y();
        let col = col >> self.chroma_shift_x();
        let row_u = self.u_row_mut(row);
        row_u[col] = value /2;
    }
    fn write_v(&mut self, row:usize, col:usize, value:u8){
        let row = row >> self.chroma_shift_y();
        let col = col >> self.chroma_shift_x();
        let row_v = self.v_row_mut(row);
        row_v[col] += value /2;
    }
    fn write_yuv(&mut self,reader:&mut dyn Iterator<Item=YUV>)->bool{
        for i in 0..self.height() as usize{
            
            self.y_row_mut(i).iter_mut().for_each(|item|*item=0u8);
            self.u_row_mut(i >> self.chroma_shift_y()).iter_mut().for_each(|item|*item=0u8);
            self.v_row_mut(i >> self.chroma_shift_y()).iter_mut().for_each(|item|*item=0u8);

            let max_bits=(1 << self.depth()) - 1;
            let max = max_bits as f32;
            for j in 0..self.width() as usize{
                let yuv = match reader.next(){
                    Some(yuv)=>yuv,
                    None=>return false
                };
                self.write_y(i, j, (yuv.y * max) as u8);
                self.write_u(i, j, (yuv.u * max) as u8);
                self.write_v(i, j, (yuv.v * max) as u8);
            }
        }
        return true;
    }
}
impl<'a> YuvCompatible for YUVImage<'a>{
    fn y_row(&self, at:usize)->&[u8]{
        return self.plane_y.row(at);
    }
    fn y_row_mut(&mut self, at:usize)->&mut [u8]{
        return self.plane_y.row_mut(at).unwrap();
    }
    fn u_row(&self, at:usize)->&[u8]{
        return self.plane_u.row(at);
    }
    fn u_row_mut(&mut self, at:usize)->&mut [u8]{
        return self.plane_u.row_mut(at).unwrap();
    }
    fn v_row(&self, at:usize)->&[u8]{
        return self.plane_v.row(at)
    }
    fn v_row_mut(&mut self, at:usize)->&mut [u8]{
        return self.plane_v.row_mut(at).unwrap();
    }
    fn chroma_shift_x(&self)->u8{
        return self.chroma_shift_x;
    }
    fn chroma_shift_y(&self)->u8{
        return self.chroma_shift_y;
    }
    fn depth(&self)->u8{
        return self.depth as u8;
    }
    fn width(&self)->usize{
        return self.width as usize;
    }
    fn height(&self)->usize{
        return self.height as usize;
    }
}
impl<'a> YUVImage<'a>{
    pub fn iter(&self)->YUVImageIter<'_>{
        YUVImageIter::new(self)
    }
    pub fn write_yuv16<Read:Iterator<Item=YUV>>(&mut self,mut reader: Read)->bool{
        for i in 0..self.height as usize{
            let row_y = self.plane_y.row_mut(i);
            let row_u = self.plane_u.row_mut(i << self.chroma_shift_y);
            let row_v = self.plane_v.row_mut(i << self.chroma_shift_y);
            let (row_y, row_u, row_v) = match (row_y, row_u, row_v){
                (Some(y), Some(u), Some(v))=>{
                    u.iter_mut().for_each(|item| *item = 0u16);
                    v.iter_mut().for_each(|item| *item = 0u16);
                    y.iter_mut().for_each(|item| *item = 0u16);
                    (y, u, v)
                },
                _=>return false
            };
            let max_bits=(1 << self.depth) - 1;
            let max = max_bits as f32;
            for j in 0..self.width as usize{
                let yuv = match reader.next(){
                    Some(yuv)=>yuv,
                    None=>return false
                };
                let u = j << self.chroma_shift_x;
                let v = j << self.chroma_shift_x;
                row_y[j] = (yuv.y * max) as u16;
                row_u[u] += (yuv.v * max) as  u16 / 2;
                row_v[v] += (yuv.u * max) as  u16 / 2;
            }
        }
        return true;
    }
} 
pub struct YUVImageIter<'a>{
    reader:&'a dyn YuvCompatible,
    index:usize
}
impl<'a> YUVImageIter<'a>{
    fn new(reader:&'a dyn YuvCompatible)->Self{
        Self{
            reader:reader,
            index:0
        }
    }
}
impl<'a> Iterator for YUVImageIter<'a>{
    type Item = YUV;
    fn next(&mut self)->Option<Self::Item>{
        if self.index >= (self.reader.height() * self.reader.width()){
            None
        }
        else{
            let row = self.index / self.reader.width();
            let col = self.index % self.reader.width();
            self.index += 1;

            let depth = ((1 << self.reader.depth()) - 1) as f32;
            
            let shift_x = self.reader.chroma_shift_x();
            let shift_y = self.reader.chroma_shift_y();
            let (y, u, v) = if self.reader.depth() > 8{
                fn to_u16_arr(arr:&[u8])->&[u16]{unsafe{
                    &std::mem::transmute::<_,&[u16]>(arr)[0..arr.len()/2]
                }}
                (
                    to_u16_arr(self.reader.y_row(row))[col] as f32,
                    to_u16_arr(self.reader.u_row(row>>shift_y))[col>>shift_x] as f32,
                    to_u16_arr(self.reader.v_row(row>>shift_y))[col>>shift_x] as f32
                )
            }
            else{
                (
                    self.reader.y_row(row)[col] as f32,
                    self.reader.u_row(row>>shift_y)[col>>shift_x] as f32,
                    self.reader.v_row(row>>shift_y)[col>>shift_x] as f32
                )
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
        self.plane_y =Some(Plane::Ref{
            data:data,
            stride:stride
        });
        return self;
    }

    pub fn plane_u(mut self, data:&'a[u8], stride:u32)->Self{
        self.plane_u =Some(Plane::Ref{
            data:data,
            stride:stride
        });
        return self;
    }
    pub fn plane_v(mut self, data:&'a[u8], stride:u32)->Self{
        self.plane_v =Some(Plane::Ref{
            data:data,
            stride:stride
        });
        return self;
    }
    pub fn plane_u_mut(mut self, data:&'a mut [u8], stride:u32)->Self{
        self.plane_u =Some(Plane::RefMut{
            data:data,
            stride:stride
        });
        return self;
    }
    pub fn plane_v_mut(mut self, data:&'a mut [u8], stride:u32)->Self{
        self.plane_v =Some(Plane::RefMut{
            data:data,
            stride:stride
        });
        return self;
    }
    pub fn plane_y_mut(mut self, data:&'a mut [u8], stride:u32)->Self{
        self.plane_y =Some(Plane::RefMut{
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
        let data =self.data.expect("data missing");
        let width= self.width.expect("width missing");
        let height=self.height.expect("height missing");
        let stride=self.stride.expect("stride missing");
        let depth=self.depth.expect("depth missing");
        RGB32Image{
            data:data,
            width:width,
            height:height,
            stride:stride,
            depth:depth
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
        let _4bytes:*const u32 = row[width_at * 4 .. (width_at + 1)*4].as_ptr() as *const u32;
        let _4bytes:u32 = unsafe{*_4bytes};
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
pub struct RGB48Writer<Reader>
where Reader:Iterator<Item=RGB>{
    rgb:Reader,
    depth:u8,
    max:f32,
    max_bits:u32
}

fn cvt_to_slice(val:u16)->[u8;2]{
    val.to_be_bytes()
}
impl<Reader> RGB48Writer<Reader>
where Reader:Iterator<Item=RGB>{
    pub fn new(reader:Reader, depth:u8)->Self{
        let max_bits=(1u32 << depth) - 1;
        let max = max_bits as f32;
        Self{
            rgb:reader,
            depth:depth,
            max:max,
            max_bits:max_bits
        }
    }
    fn get(&self, rgb:RGB)->[u8;6]{
        let max = self.max;
        let r:[u8;2] = cvt_to_slice((rgb.r * max) as u16);
        let g:[u8;2] = cvt_to_slice((rgb.g * max) as u16);
        let b:[u8;2] = cvt_to_slice((rgb.b * max) as u16);
        return [r[0], r[1],g[0],g[1],b[0],b[1]];
    }
}
impl<Reader> Iterator for RGB48Writer<Reader>
where Reader:Iterator<Item=RGB>{
    type Item = [u8;6];
    fn next(&mut self)->Option<Self::Item>{
        let rgb = self.rgb.next()?;
        return Some(self.get(rgb));
    }
}
impl<Reader> std::io::Read for RGB48Writer<Reader>
where Reader:Iterator<Item=RGB>{
    fn read(&mut self, buf:&mut [u8])->Result<usize, std::io::Error>{
        let mut offset = 0usize;

        while buf.len() - offset >= 6{
            match self.rgb.next(){
                None if offset == 0 =>return Err(std::io::Error::from(std::io::ErrorKind::WriteZero)),
                None if offset != 0 =>return Ok(offset),
                Some(rgb)=>{
                    let rgb = self.get(rgb);
                    buf[offset..offset+6].copy_from_slice(&rgb);
                    offset += 6;
                }
                _=>unreachable!()
            }    
        }
        return Ok(offset);
    }
    
}
use std::io::Read;
struct RGB24Reader<T:Read>{
    data:T
}
impl<T:Read> RGB24Reader<T>{
    fn new(data:T)->Self{
        Self{
            data:data
        }
    }
}
impl<T:Read> Iterator for RGB24Reader<T>{
    type Item = RGB;
    fn next(&mut self)->Option<Self::Item>{
        let mut rgb_bytes = [0u8;3];
        let mut total_read_count = 0;
        while let Ok(read_count) = self.data.read(&mut rgb_bytes[total_read_count..]){
            total_read_count += read_count;
            if read_count == 0 || total_read_count == 3{
                break;
            }
        }
        return if total_read_count != 3{
            None
        }
        else{
            let r = rgb_bytes[0] as f32 / 255f32;
            let g = rgb_bytes[1] as f32 / 255f32;
            let b = rgb_bytes[2] as f32 / 255f32;
            let rgb = RGB::new(r,g,b);
            Some(rgb)
        };
    }
}

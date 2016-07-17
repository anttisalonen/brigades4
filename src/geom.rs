#[derive(Copy, Clone)]
pub struct Vertex {
        pub position: (f32, f32, f32)
}

implement_vertex!(Vertex, position);

#[derive(Copy, Clone)]
pub struct Normal {
    pub normal: (f32, f32, f32)
}

implement_vertex!(Normal, normal);

pub struct Geom {
    pub vertices: Vec<Vertex>,
    pub normals: Vec<Normal>,
    pub indices: Vec<u16>,
}

pub fn new_geom(num_vertices: usize, num_indices: usize) -> Geom {
    Geom {
        vertices: vec![Vertex{position: (0.0, 0.0, 0.0)}; num_vertices],
        normals: vec![Normal{normal: (0.0, 0.0, 0.0)}; num_vertices],
        indices: vec![0; num_indices],
    }
}

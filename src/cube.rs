use geom;

pub const VERTICES: [geom::Vertex; 8] = [
    geom::Vertex { position: (-0.5, -0.5, -0.5) },
    geom::Vertex { position: (0.5,  -0.5, -0.5) },
    geom::Vertex { position: (-0.5,  0.5, -0.5) },
    geom::Vertex { position: (0.5,   0.5, -0.5) },
    geom::Vertex { position: (-0.5, -0.5,  0.5) },
    geom::Vertex { position: (0.5,  -0.5,  0.5) },
    geom::Vertex { position: (-0.5,  0.5,  0.5) },
    geom::Vertex { position: (0.5,   0.5,  0.5) },
];

pub const NORMALS: [geom::Normal; 8] = [
    geom::Normal { normal: (-0.5,-0.5, 0.5) },
    geom::Normal { normal: ( 0.5,-0.5, 0.5) },
    geom::Normal { normal: (-0.5, 0.5, 0.5) },
    geom::Normal { normal: ( 0.5, 0.5, 0.5) },
    geom::Normal { normal: (-0.5,-0.5,-0.5) },
    geom::Normal { normal: ( 0.5,-0.5,-0.5) },
    geom::Normal { normal: (-0.5, 0.5,-0.5) },
    geom::Normal { normal: ( 0.5, 0.5,-0.5) },
];

pub const INDICES: [u16; 36] = [
    1, 0, 2, // front
    3, 1, 2,
    4, 5, 6, // back
    5, 7, 6,
    0, 4, 2, // left
    4, 6, 2,
    5, 1, 3, // right
    7, 5, 3,
    6, 7, 2, // top
    7, 3, 2,
    5, 4, 0, // bottom
    1, 5, 0,
];

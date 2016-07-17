#[macro_use]
extern crate glium;
extern crate nalgebra as na;
extern crate time;

mod geom;
mod cube;
mod ai;
mod game;

use na::{Vector3, Norm};
use game::{Soldier, GROUND_SIZE, TILE_SIZE, GameState};

fn soldier_model_matrix(s: &Soldier) -> [[f32; 4]; 4] {
    return [
        [f32::cos(s.direction), 0.0, f32::sin(s.direction), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-f32::sin(s.direction), 0.0, f32::cos(s.direction), 0.0],
        [s.position.x, s.position.y, s.position.z, 1.0f32]
    ];
}

const WATER_VERTICES: [geom::Vertex; 4] = [
    geom::Vertex { position: (0.0, 0.0, 0.0) },
    geom::Vertex { position: (GROUND_SIZE as f32 * TILE_SIZE, 0.0, 0.0) },
    geom::Vertex { position: (0.0, 0.0, GROUND_SIZE as f32 * TILE_SIZE) },
    geom::Vertex { position: (GROUND_SIZE as f32 * TILE_SIZE, 0.0, GROUND_SIZE as f32 * TILE_SIZE) },
];

const WATER_NORMALS: [geom::Normal; 2] = [
    geom::Normal { normal: ( 0.0, 1.0, 0.0) },
    geom::Normal { normal: ( 0.0, 1.0, 0.0) },
];

const WATER_INDICES: [u16; 6] = [
    1, 0, 2,
    3, 1, 2,
];

fn main() {
    use glium::{DisplayBuild, Surface};
    let display = glium::glutin::WindowBuilder::new()
                        .with_depth_buffer(24)
                        .build_glium().unwrap();

    let positions = glium::VertexBuffer::new(&display, &cube::VERTICES).unwrap();
    let normals = glium::VertexBuffer::new(&display, &cube::NORMALS).unwrap();
    let indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList,
                                          &cube::INDICES).unwrap();

    let vertex_shader_src = r#"
        #version 140

        in vec3 position;
        in vec3 normal;

        out vec3 v_normal;

        uniform mat4 perspective;
        uniform mat4 view;
        uniform mat4 model;

        void main() {
            mat4 modelview = view * model;
            v_normal = normal;
            gl_Position = perspective * modelview * vec4(position, 1.0);
        }
    "#;

    let fragment_shader_src = r#"
        #version 140

        in vec3 v_normal;
        out vec4 color;
        uniform vec3 u_light;
        uniform vec3 u_color;

        void main() {
            float brightness = dot(normalize(v_normal), normalize(-u_light)) + 0.2;
            vec3 dark_color = vec3(0.0, 0.0, 0.0);
            color = vec4(mix(dark_color, u_color, brightness), 1.0);
        }
    "#;

    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src,
                                              None).unwrap();

    let mut game_state = GameState::new(display);

    let ground_geom = game::get_ground_geometry(&game_state.bf.ground);
    let ground_positions = glium::VertexBuffer::new(&game_state.bf.display, &ground_geom.vertices).unwrap();
    let ground_normals = glium::VertexBuffer::new(&game_state.bf.display, &ground_geom.normals).unwrap();
    let ground_indices = glium::IndexBuffer::new(&game_state.bf.display, glium::index::PrimitiveType::TrianglesList,
                                          &ground_geom.indices).unwrap();

    let water_positions = glium::VertexBuffer::new(&game_state.bf.display, &WATER_VERTICES).unwrap();
    let water_normals = glium::VertexBuffer::new(&game_state.bf.display, &WATER_NORMALS).unwrap();
    let water_indices = glium::IndexBuffer::new(&game_state.bf.display, glium::index::PrimitiveType::TrianglesList,
                                          &WATER_INDICES).unwrap();


    let mut prev_time = time::precise_time_ns();
    loop {
        let curr_time = time::precise_time_ns();
        let frame_time = (curr_time - prev_time) as f64 / 1000000000.0;
        prev_time = curr_time;

        if ! game::update_game_state(&mut game_state, frame_time) {
            break;
        }

        if let Some(winner) = game::won(&game_state) {
            println!("Winner: {} team", if winner { "Red" } else { "Blue" });
            break;
        }

        let mut target = game_state.bf.display.draw();
        target.clear_color_and_depth((0.5, 0.5, 1.0, 1.0), 1.0);

        let view = view_matrix(&game_state.bf.camera.position,
                               &game_state.bf.camera.direction,
                               &game_state.bf.camera.upvec);

        let perspective = {
            let (width, height) = target.get_dimensions();
            let aspect_ratio = height as f32 / width as f32;

            let fov: f32 = 3.141592 / 3.0;
            let zfar = 1024.0;
            let znear = 0.1;

            let f = 1.0 / (fov / 2.0).tan();

            [
                [f *   aspect_ratio   ,    0.0,              0.0              ,   0.0],
                [         0.0         ,     f ,              0.0              ,   0.0],
                [         0.0         ,    0.0,  (zfar+znear)/(zfar-znear)    ,   1.0],
                [         0.0         ,    0.0, -(2.0*zfar*znear)/(zfar-znear),   0.0],
            ]
        };

        let light = [f32::cos(game_state.bf.curr_time as f32 * 0.5) * 0.5, -1.0, f32::sin(game_state.bf.curr_time as f32 * 0.5) * 0.5f32];
        let identity_matrix: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ];

        let params = glium::DrawParameters {
            depth: glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                .. Default::default()
            },
            backface_culling: glium::draw_parameters::BackfaceCullingMode::CullCounterClockwise,
            .. Default::default()
        };

        target.draw((&water_positions, &water_normals), &water_indices, &program,
                    &uniform! { model: identity_matrix, view: view, perspective: perspective,
                    u_light: light, u_color: [0.0, 0.0, 0.9f32] },
                    &params).unwrap();

        target.draw((&ground_positions, &ground_normals), &ground_indices, &program,
                    &uniform! { model: identity_matrix, view: view, perspective: perspective,
                    u_light: light, u_color: [0.2, 0.8, 0.2f32] },
                    &params).unwrap();
        for sold in game_state.bf.soldiers.iter() {
            let col = if sold.alive {
                if sold.side {
                    [1.0, 0.0, 0.0f32]
                } else {
                    [0.0, 0.0, 1.0f32]
                }
            } else {
                [0.0, 0.0, 0.0f32]
            };
            target.draw((&positions, &normals), &indices, &program,
                        &uniform! { model: soldier_model_matrix(&sold), view: view, perspective: perspective,
                        u_light: light, u_color: col },
                        &params).unwrap();
        }
        target.finish().unwrap();
    }
}


fn view_matrix(position: &Vector3<f32>, direction: &Vector3<f32>, up: &Vector3<f32>) -> [[f32; 4]; 4] {
    let f = direction.normalize();

    let s = [up.y * f.z - up.z * f.y,
             up.z * f.x - up.x * f.z,
             up.x * f.y - up.y * f.x];

    let s_norm = {
        let len = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
        let len = len.sqrt();
        [s[0] / len, s[1] / len, s[2] / len]
    };

    let u = [f.y * s_norm[2] - f.z * s_norm[1],
             f.z * s_norm[0] - f.x * s_norm[2],
             f.x * s_norm[1] - f.y * s_norm[0]];

    let p = [-position.x * s_norm[0] - position.y * s_norm[1] - position.z * s_norm[2],
             -position.x * u[0] - position.y * u[1] - position.z * u[2],
             -position.x * f.x - position.y * f.y - position.z * f.z];

    [
        [s_norm[0], u[0], f.x, 0.0],
        [s_norm[1], u[1], f.y, 0.0],
        [s_norm[2], u[2], f.z, 0.0],
        [p[0], p[1], p[2], 1.0],
    ]
}


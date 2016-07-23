#[macro_use]
extern crate glium;
extern crate nalgebra as na;
extern crate time;
extern crate image;

mod geom;
mod cube;
mod ai;
mod game;
mod gameutil;

use na::{Vector3, Norm};
use game::{Soldier, GROUND_NUM_TILES, TILE_SIZE, GameState};

const WATER_MODEL_MATRIX: [[f32; 4]; 4] = {
    const DIM: f32 = GROUND_NUM_TILES as f32 * TILE_SIZE;
    [
        [DIM, 0.0, 0.0, 0.0],
        [0.0, DIM, 0.0, 0.0],
        [0.0, 0.0, DIM, 0.0],
        [0.0, 0.0, 0.0, 1.0f32],
    ]
};

fn icon_model_matrix(pos: &Vector3<f32>) -> [[f32; 4]; 4] {
    let yp = f32::max(pos.y, 0.0) + 10.0;
    [
        [24.0, 0.0,  0.0,  0.0],
        [0.0,  24.0, 0.0,  0.0],
        [0.0,  0.0,  24.0, 0.0],
        [pos.x, f32::max(yp, 0.0) + 10.0, pos.z, 1.0f32]
    ]
}

fn flag_color(flag: &game::Flag) -> [f32; 3] {
    match flag.flag_state {
        game::FlagState::Free                         => [0.8, 0.8, 0.8],
        game::FlagState::Transition(game::Side::Blue) => [0.2, 0.2, 0.6],
        game::FlagState::Transition(game::Side::Red)  => [0.6, 0.2, 0.2],
        game::FlagState::Owned(game::Side::Blue)      => [0.0, 0.0, 1.0],
        game::FlagState::Owned(game::Side::Red)       => [1.0, 0.0, 0.0],
    }
}

fn soldier_model_matrix(s: &Soldier) -> [[f32; 4]; 4] {
    [
        [f32::cos(s.direction), 0.0, f32::sin(s.direction), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-f32::sin(s.direction), 0.0, f32::cos(s.direction), 0.0],
        [s.position.x, s.position.y, s.position.z, 1.0f32]
    ]
}

const WATER_VERTICES: [geom::Vertex; 4] = [
    geom::Vertex { position: (0.0, 0.0, 0.0) },
    geom::Vertex { position: (1.0, 0.0, 0.0) },
    geom::Vertex { position: (0.0, 0.0, 1.0) },
    geom::Vertex { position: (1.0, 0.0, 1.0) },
];

const FLAG_VERTICES: [geom::TexVertex; 4] = [
    geom::TexVertex { position: (0.0, 0.0, 0.0), tex_coords: (0.0, 0.0) },
    geom::TexVertex { position: (1.0, 0.0, 0.0), tex_coords: (1.0, 0.0) },
    geom::TexVertex { position: (0.0, 0.0, 1.0), tex_coords: (0.0, 1.0) },
    geom::TexVertex { position: (1.0, 0.0, 1.0), tex_coords: (1.0, 1.0) },
];

const WATER_NORMALS: [geom::Normal; 2] = [
    geom::Normal { normal: ( 0.0, 1.0, 0.0) },
    geom::Normal { normal: ( 0.0, 1.0, 0.0) },
];

const WATER_INDICES: [u16; 6] = [
    1, 0, 2,
    3, 1, 2,
];

const IDENTITY_MATRIX: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
];

struct Gfx {
    icon_positions: glium::VertexBuffer<geom::TexVertex>,
    water_normals: glium::VertexBuffer<geom::Normal>,
    water_indices: glium::IndexBuffer<u16>,
    program: glium::Program,
    icon_program: glium::Program,
}

struct GfxPerFrame {
    target: glium::Frame,
    perspective: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    light: [f32; 3],
    ambient: f32,
}

fn main() {
    use glium::{DisplayBuild, Surface};
    let display = glium::glutin::WindowBuilder::new()
                        .with_depth_buffer(24)
                        .build_glium().unwrap();

    let flag_texture = load_texture("share/flag.png", &display);
    let food_texture = load_texture("share/food.png", &display);

    let positions = glium::VertexBuffer::new(&display, &cube::VERTICES).unwrap();
    let normals = glium::VertexBuffer::new(&display, &cube::NORMALS).unwrap();
    let indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList,
                                          &cube::INDICES).unwrap();

    let vertex_shader_src = r#"
        #version 130

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
        #version 130

        in vec3 v_normal;
        out vec4 color;
        uniform vec3 u_light;
        uniform float u_ambient;
        uniform vec3 u_color;

        void main() {
            float brightness = max(dot(normalize(v_normal), u_light), u_ambient);
            vec3 dark_color = vec3(0.0, 0.0, 0.0);
            color = vec4(mix(dark_color, u_color, brightness), 1.0);
        }
    "#;

    let icon_vertex_shader_src = r#"
        #version 130

        in vec3 position;
        in vec3 normal;
        in vec2 tex_coords;

        out vec3 v_normal;
        out vec2 v_texcoord;

        uniform mat4 perspective;
        uniform mat4 view;
        uniform mat4 model;

        void main() {
            mat4 modelview = view * model;
            v_normal = normal;
            v_texcoord = tex_coords;
            gl_Position = perspective * modelview * vec4(position, 1.0);
        }
    "#;

    let icon_fragment_shader_src = r#"
        #version 130

        in vec3 v_normal;
        in vec2 v_texcoord;
        out vec4 color;
        uniform vec3 u_light;
        uniform float u_ambient;
        uniform vec3 u_color;
        uniform sampler2D tex;

        void main() {
            vec4 tex_value = texture(tex, v_texcoord);
            float brightness = max(dot(normalize(v_normal), u_light), u_ambient);
            if(tex_value.a < 0.5)
                discard;

            vec3 dark_color = vec3(0.0, 0.0, 0.0);
            color = vec4(mix(dark_color, u_color, brightness), 1.0) * tex_value;
        }
    "#;

    let icon_program = glium::Program::from_source(&display, icon_vertex_shader_src, icon_fragment_shader_src,
                                              None).unwrap();

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

    let icon_positions = glium::VertexBuffer::new(&game_state.bf.display, &FLAG_VERTICES).unwrap();

    let params = glium::DrawParameters {
        depth: glium::Depth {
            test: glium::draw_parameters::DepthTest::IfLess,
            write: true,
            .. Default::default()
        },
        backface_culling: glium::draw_parameters::BackfaceCullingMode::CullCounterClockwise,
        .. Default::default()
    };

    let icon_params = glium::DrawParameters {
        depth: glium::Depth {
            test: glium::draw_parameters::DepthTest::IfLess,
            write: true,
            .. Default::default()
        },
        .. Default::default()
    };

    let gfx = Gfx {
        icon_positions: icon_positions,
        water_normals: water_normals,
        water_indices: water_indices,
        program: program,
        icon_program: icon_program,
    };

    let mut prev_time = time::precise_time_ns();
    loop {
        let curr_time = time::precise_time_ns();
        let frame_time = (curr_time - prev_time) as f64 / 1000000000.0;
        prev_time = curr_time;

        if ! game::update_game_state(&mut game_state, frame_time) {
            break;
        }

        if let Some(winner) = game::won(&game_state) {
            println!("Winner: {} team", if winner == game::Side::Blue { "Blue" } else { "Red" });
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

        // curr_day_time is between 0 and 2pi
        // day_time is between -1 and 1
        let curr_day_time = game::curr_day_time(&game_state) * 2.0 * 3.141592;
        let day_time = -f32::cos(curr_day_time);
        let ambient = 0.01f32;
        let light_len = gameutil::clamp(0.0, 1.0, day_time * 2.0 + 1.0);
        let light = Vector3::<f32>::new(f32::sin(curr_day_time), light_len, -f32::sin(curr_day_time * 0.5)).normalize();
        let light = gameutil::truncate(light * light_len, 1.0);

        let mut gfx_per_frame = GfxPerFrame {
            target: target,
            perspective: perspective,
            view: view,
            light: [light.x, light.y, light.z],
            ambient: ambient,
        };

        // flag
        for flag in game_state.bf.flags.iter() {
            draw_icon(&gfx, &mut gfx_per_frame, &flag.position,
                      flag_color(&flag), &flag_texture, &icon_params);
        }

        // supply
        for supply in game_state.bf.supply_points.iter() {
            draw_icon(&gfx, &mut gfx_per_frame, &supply.position,
                      [1.0, 1.0, 1.0], &food_texture, &icon_params);
        }


        // water
        draw_model(&gfx, &mut gfx_per_frame, &water_positions, &gfx.water_normals, &gfx.water_indices,
                   &WATER_MODEL_MATRIX, [0.0, 0.0, 0.9f32], &params);

        // ground
        draw_model(&gfx, &mut gfx_per_frame, &ground_positions, &ground_normals, &ground_indices,
                   &IDENTITY_MATRIX, [0.2, 0.8, 0.2f32], &params);

        for sold in game_state.bf.soldiers.iter() {
            let col = if sold.alive {
                if sold.side == game::Side::Red {
                    [1.0, 0.0, 0.0f32]
                } else {
                    [0.0, 0.0, 1.0f32]
                }
            } else {
                [0.0, 0.0, 0.0f32]
            };
            draw_model(&gfx, &mut gfx_per_frame, &positions, &normals, &indices,
                       &soldier_model_matrix(&sold), col, &params);
        }
        gfx_per_frame.target.finish().unwrap();
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

fn draw_icon(gfx: &Gfx, mut gfx_per_frame: &mut GfxPerFrame,
             position: &Vector3<f32>,
             color: [f32; 3], texture: &glium::Texture2d, params: &glium::DrawParameters) -> () {
    use glium::{Surface};
    gfx_per_frame.target.draw((&gfx.icon_positions, &gfx.water_normals), &gfx.water_indices, &gfx.icon_program,
        &uniform! {
            model: icon_model_matrix(&position),
            view: gfx_per_frame.view,
            perspective: gfx_per_frame.perspective,
            u_light: gfx_per_frame.light,
            u_ambient: gfx_per_frame.ambient,
            u_color: color,
            tex: texture,
        },
        &params).unwrap();
}

fn draw_model(gfx: &Gfx, mut gfx_per_frame: &mut GfxPerFrame,
    positions: &glium::VertexBuffer<geom::Vertex>,
    normals: &glium::VertexBuffer<geom::Normal>,
    indices: &glium::IndexBuffer<u16>,
    model: &[[f32; 4]; 4],
    color: [f32; 3], params: &glium::DrawParameters) -> () {
    use glium::{Surface};
    gfx_per_frame.target.draw((positions, normals), indices, &gfx.program,
        &uniform! {
            model: *model,
            view: gfx_per_frame.view,
            perspective: gfx_per_frame.perspective,
            u_light: gfx_per_frame.light,
            u_ambient: gfx_per_frame.ambient,
            u_color: color
        },
        &params).unwrap();
}

fn load_texture(filename: &str, display: &glium::Display) -> glium::Texture2d {
    use std::path::Path;
    let img = image::open(&Path::new(filename)).unwrap().to_rgba();
    let img_dimensions = img.dimensions();
    let img = glium::texture::RawImage2d::from_raw_rgba_reversed(img.into_raw(), img_dimensions);
    glium::texture::Texture2d::new(display, img).unwrap()
}


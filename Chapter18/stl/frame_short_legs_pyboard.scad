// frame for short four-legged robot (dimensions for pyboard hardware)
// History:
// v1: Fusion360, check 01_trivial dir
// v2: openscad: 
//     - inverted servos support
//     - solid holder for IMU unit

// servo motor GH-S37A
SERVO_WIDTH = 13;
SERVO_LENGTH = 20.1;
SERVO_THICK = 9;
SERVO_EARS_LENGTH = 5.5;
SERVO_EARS_THICK = 1;
SERVO_HOLE_LENGTH = 7.5;
SERVO_HOLE_D = 1.5;
SERVO_WIRE_THICK = 2.5;
SERVO_WIRE_LENGTH = 10;

module servo(up_extrude = 1) {
    cube([SERVO_WIDTH, SERVO_LENGTH, SERVO_THICK*up_extrude]);
    translate([-SERVO_EARS_THICK, -SERVO_EARS_LENGTH, 0])
    cube([SERVO_EARS_THICK, (SERVO_EARS_LENGTH+0.6)*2 + SERVO_LENGTH, SERVO_THICK*up_extrude]);

    translate([-SERVO_EARS_THICK*2, -(SERVO_EARS_LENGTH)/2, SERVO_THICK/2])
    rotate([0, 90, 0])
    cylinder(SERVO_HOLE_LENGTH + SERVO_EARS_THICK*2, d=SERVO_HOLE_D, $fn=10);

    translate([-SERVO_EARS_THICK*2, SERVO_LENGTH + SERVO_EARS_LENGTH/2, SERVO_THICK/2])
    rotate([0, 90, 0])
    cylinder(SERVO_HOLE_LENGTH + SERVO_EARS_THICK*2, d=SERVO_HOLE_D, $fn=10);
    
    // cable
    translate([SERVO_WIDTH-SERVO_WIRE_THICK+0.7, SERVO_WIRE_THICK/2+.5, SERVO_THICK/3])
    rotate([0, 0, -45])
    cube([SERVO_WIRE_THICK, SERVO_WIRE_LENGTH, SERVO_THICK*2]);
}


BOARD_LENGTH = 50;
BOARD_WIDTH = 33;
BOARD_THICK = 1.6 + 2;
BOARD_HOLE_OFS = 2;

module board(up_extrude = 1) {
    cube([BOARD_WIDTH, BOARD_LENGTH, BOARD_THICK*up_extrude]);
}

// Frame base with servo and board placement
BASE_X_BOARD_SPACE = 5;
BASE_SERVO_DIST = 1;
BASE_LENGTH = (SERVO_LENGTH+2*SERVO_EARS_LENGTH+BASE_SERVO_DIST)*2;
BASE_WIDTH = BOARD_WIDTH + BASE_X_BOARD_SPACE*2;
BASE_THICK = 10;
BASE_UNDER_SERVO_DIM = 1;
BASE_BOARD_Z_OFS = BASE_THICK + 2;
BASE_INNER_THICK = 2;
BOARD_SCREW_DEPTH = 5;
BOARD_SCREW_D = 1.5;

HOLDER_LENGTH = 3;
HOLDER_EXTRA = 2;
HOLDER_THICK = BOARD_THICK + HOLDER_EXTRA;


//                translate([0, BOARD_LENGTH, BASE_THICK])
//                cube([BASE_WIDTH, BOARD_HOLDER_LENGTH, BOARD_HOLDER_THICK]);

module board_holder() {
    rotate([90, 0, 90])
    linear_extrude(BASE_WIDTH)
    polygon(points=[
        [0, 0], [HOLDER_LENGTH, 0], [HOLDER_LENGTH, HOLDER_THICK],
        [-HOLDER_EXTRA, HOLDER_THICK], [0, HOLDER_THICK - HOLDER_EXTRA]
    ]);
}


module frame_base() {
    difference() {
        union() {
            cube([BASE_WIDTH, BASE_LENGTH, BASE_THICK]);
            
            // board holder
            translate([0, BOARD_LENGTH, BASE_THICK])
            board_holder();
        }
        
        translate([SERVO_EARS_THICK, SERVO_EARS_LENGTH, BASE_UNDER_SERVO_DIM])
        servo(up_extrude=3);
        
        translate([SERVO_EARS_THICK, BASE_LENGTH-SERVO_EARS_LENGTH, BASE_UNDER_SERVO_DIM])
        mirror([0, 1, 0])
        servo(up_extrude=3);

        translate([BASE_WIDTH, 0, 0])
        mirror([1, 0, 0]) {
            translate([SERVO_EARS_THICK, SERVO_EARS_LENGTH, BASE_UNDER_SERVO_DIM])
            servo(up_extrude=3);
            mirror([0, 1, 0])
            translate([SERVO_EARS_THICK, -BASE_LENGTH+SERVO_EARS_LENGTH, BASE_UNDER_SERVO_DIM])
            servo(up_extrude=3);
        }        
        
        translate([SERVO_WIDTH + SERVO_EARS_THICK + BASE_INNER_THICK, BASE_INNER_THICK, 0])
        cube([BASE_WIDTH - SERVO_WIDTH*2 - SERVO_EARS_THICK*2 - BASE_INNER_THICK*2, 
              BASE_LENGTH - BASE_INNER_THICK*2, BASE_THICK + HOLDER_THICK]);

        // board screw holes
        translate([BASE_X_BOARD_SPACE, 0, BASE_THICK - BOARD_SCREW_DEPTH]) {
            translate([BOARD_HOLE_OFS, BOARD_HOLE_OFS, 0])
            cylinder(BOARD_SCREW_DEPTH, d=BOARD_SCREW_D, $fn=10);
            translate([BOARD_WIDTH-BOARD_HOLE_OFS, BOARD_HOLE_OFS, 0])
            cylinder(BOARD_SCREW_DEPTH, d=BOARD_SCREW_D, $fn=10);
        }
    }
        
    
}

module imu_holder(thickness) {
    // grip offsets
    linear_extrude(thickness)
    polygon([
        [4.3, 0], 
        [4.3, 10],
        [2.0, 10], [2.0, 12], 
        [4.3, 14], [0.0, 14],
        [0.0, 0], 
    ]);
}

IMU_WIDTH = 25.4;
IMU_LENGTH = 25.6;
IMU_THICK = 1.7;

module imu_board() {
    x_ofs = (BASE_WIDTH - IMU_WIDTH)/2;
    y_ofs = (BASE_LENGTH - IMU_LENGTH)/2;
    
//    translate([x_ofs, y_ofs, 34])
    translate([2, 0, 0])
    cube([IMU_WIDTH, IMU_LENGTH, IMU_THICK]);
}


module imu_holders() {
    translate([IMU_WIDTH+4, 0, 0])
    mirror([1, 0, 0])
    rotate([90, 0, 0])
    imu_holder(SERVO_EARS_LENGTH);

    rotate([90, 0, 0])
    imu_holder(SERVO_EARS_LENGTH);
}

//imu_board();

//servo();
frame_base();

translate([(BASE_WIDTH - IMU_WIDTH-4)/2, BASE_LENGTH, BASE_THICK])
imu_holders();

//translate([BASE_X_BOARD_SPACE, 0, BASE_THICK])
//board();

//board_holder();

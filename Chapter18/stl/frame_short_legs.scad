    // frame for short four-legged robot
    // History:
    // v1: Fusion360, check 01_trivial dir
    // v2: openscad: 
    //     - inverted servos support
    //     - solid holder for IMU unit
    
    // servo motor GH-S37A
    SERVO_WIDTH = 13;
    SERVO_LENGTH = 20;
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
        cube([SERVO_EARS_THICK, (SERVO_EARS_LENGTH+0.5)*2 + SERVO_LENGTH, SERVO_THICK*up_extrude]);
    
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
    
    
    BOARD_LENGTH = 60;
    BOARD_WIDTH = 37;
    BOARD_THICK = 2;
    BOARD_HOLE_D = 2;
    BOARD_HOLE_OFS = 4;
    BOARD_HOLE_LENGTH = 10;
    BOARD_HOLE_BASE = 8;
    
    BOARD_DOWN_LENGTH = 52;
    BOARD_DOWN_WIDTH = 33;
    BOARD_DOWN_DEPTH = 17;
    
    BOARD_SD_WIDTH = 15;
    BOARD_SD_LENGTH = 18;
    BOARD_SD_THICK = 3;
    BOARD_SD_Y_OFS = 8;
    
    module board(up_extrude = 1) {
        difference() {
            union() {
                cube([BOARD_WIDTH, BOARD_LENGTH, BOARD_THICK*up_extrude]);
                translate([(BOARD_WIDTH - BOARD_DOWN_WIDTH)/2, 
                    (BOARD_LENGTH - BOARD_DOWN_LENGTH)/2, -BOARD_DOWN_DEPTH])
                cube([BOARD_DOWN_WIDTH, BOARD_DOWN_LENGTH, BOARD_DOWN_DEPTH]);
            }
            
            translate([0, 0, -BOARD_DOWN_DEPTH]) {
                cube([BOARD_HOLE_BASE, BOARD_HOLE_BASE, BOARD_DOWN_DEPTH]);
                
                translate([BOARD_WIDTH - BOARD_HOLE_BASE, 0, 0])
                cube([BOARD_HOLE_BASE, BOARD_HOLE_BASE, BOARD_DOWN_DEPTH]);
    
                translate([0, BOARD_LENGTH - BOARD_HOLE_BASE, 0])
                cube([BOARD_HOLE_BASE, BOARD_HOLE_BASE, BOARD_DOWN_DEPTH]);
    
                translate([BOARD_WIDTH - BOARD_HOLE_BASE, BOARD_LENGTH - BOARD_HOLE_BASE, 0])
                cube([BOARD_HOLE_BASE, BOARD_HOLE_BASE, BOARD_DOWN_DEPTH]);
            }
        }
    
        translate([0, BOARD_SD_Y_OFS, -BOARD_THICK])
        cube([BOARD_SD_WIDTH, BOARD_SD_LENGTH, BOARD_SD_THICK]);
        
        translate([BOARD_HOLE_OFS, BOARD_HOLE_OFS, -BOARD_HOLE_LENGTH+BOARD_THICK*2])
        cylinder(BOARD_HOLE_LENGTH, d=BOARD_HOLE_D, $fn=10);
    
        translate([BOARD_WIDTH-BOARD_HOLE_OFS, BOARD_HOLE_OFS, -BOARD_HOLE_LENGTH+BOARD_THICK*2])
        cylinder(BOARD_HOLE_LENGTH, d=BOARD_HOLE_D, $fn=10);
    
        translate([BOARD_HOLE_OFS, BOARD_LENGTH-BOARD_HOLE_OFS, -BOARD_HOLE_LENGTH+BOARD_THICK*2])
        cylinder(BOARD_HOLE_LENGTH, d=BOARD_HOLE_D, $fn=10);
    
        translate([BOARD_WIDTH-BOARD_HOLE_OFS, BOARD_LENGTH-BOARD_HOLE_OFS, -BOARD_HOLE_LENGTH+BOARD_THICK*2])
        cylinder(BOARD_HOLE_LENGTH, d=BOARD_HOLE_D, $fn=10);
    }
    
    
    
    // Frame base with servo and board placement
    BASE_LENGTH = 63;
    BASE_WIDTH = 66;
    BASE_THICK = 10;
    BASE_UNDER_SERVO_DIM = 0;
    BASE_BOARD_Z_OFS = BASE_THICK + 2;
    
    BASE_WIRE_LENGTH = 6;
    BASE_WIRE_THICK = 1;
    
    
    module frame_base() {
        board_ofs_x = (BASE_WIDTH - BOARD_WIDTH)/2;
        board_ofs_y = (BASE_LENGTH - BOARD_LENGTH)/2;
        
        difference() {
            union() {
                cube([BASE_WIDTH, BASE_LENGTH, BASE_THICK]);
                translate([board_ofs_x, board_ofs_y, 0])
                cube([BOARD_WIDTH, BOARD_LENGTH, BASE_BOARD_Z_OFS]);
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
    
            // board
            translate([board_ofs_x, board_ofs_y, BASE_BOARD_Z_OFS])
            board(1);
            
            // wire hole
            translate([(BASE_WIDTH - SERVO_WIRE_THICK)/2, -1, BASE_THICK*2/3])
            cube([SERVO_WIRE_THICK, 10, BASE_THICK]);
        }
        
        // wire supports
        translate([board_ofs_x, (BASE_LENGTH-BASE_WIRE_LENGTH)/3, 0])
        cube([BOARD_WIDTH, BASE_WIRE_LENGTH, BASE_WIRE_THICK]);
    
        translate([board_ofs_x, (BASE_LENGTH-BASE_WIRE_LENGTH)*2/3, 0])
        cube([BOARD_WIDTH, BASE_WIRE_LENGTH, BASE_WIRE_THICK]);
    }
    
    
    module imu_holder(thickness) {
        // grip offsets
        y = 5;
        x = 1.5;
        linear_extrude(thickness)
        translate([5, 14])
        polygon([
            [1, -2],
            [1, -14], [6, -14],
            [6, -1], [17.3-x, 14-y], [17.3-x, 15-y], [15-x, 15-y],
            [15-x, 17-y], [17.3-x, 19-y], [13-x, 19-y], [13-x, 13-y]
        ]);
    }
    
    
    FRAME_IMU_THICK = 6;
    FRAME_IMU_X_OFS = 2;
    
    
    module frame() {
        frame_base();
        
        imu_y_ofs = (BASE_LENGTH + FRAME_IMU_THICK)/2;
        
        translate([FRAME_IMU_X_OFS, imu_y_ofs, BASE_THICK])
        rotate([90, 0, 0])
        imu_holder(FRAME_IMU_THICK);
    
        translate([BASE_WIDTH - FRAME_IMU_X_OFS, imu_y_ofs, BASE_THICK])
        mirror([1, 0, 0])
        rotate([90, 0, 0])
        imu_holder(FRAME_IMU_THICK);
    }
    
    
    IMU_WIDTH = 25.4;
    IMU_LENGTH = 25.6;
    IMU_THICK = 1.7;
    
    module imu_board() {
        x_ofs = (BASE_WIDTH - IMU_WIDTH)/2;
        y_ofs = (BASE_LENGTH - IMU_LENGTH)/2;
        
        translate([x_ofs, y_ofs, 34])
        cube([IMU_WIDTH, IMU_LENGTH, IMU_THICK]);
    }
    
    
    //frame_base();
    //imu_holder(FRAME_IMU_THICK);
    //board();
    //servo();
    frame();
    
    //imu_board();
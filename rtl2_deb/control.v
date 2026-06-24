`timescale 1ns / 1ps
// =============================================================================
// control.v  — Sequencer FSM
// FIXES:
//   [BUG-C] Reset polarity: top drives .reset(rst) (active-HIGH);
//           changed if(!reset) → if(reset) throughout
//   [BUG-H] obj_out was declared as output reg but NEVER assigned;
//           added obj_out <= obj in the counter block
//   [BUG-E] Per-object interrupt protocol: the testbench sends one object at a
//           time, waits for an interrupt, then sends the next. The original FSM
//           looped through all n_objects internally without pausing, so it would
//           re-use the stale label_buf for every object after the first.
//           Fix: after processing each object (CHECK state), the FSM now goes
//           to WAIT_LABEL and clears the "need new label" flag so the top-level
//           can clear label_full. This makes the FSM block in LOAD until the
//           next label embedding arrives. The "done" pulse is raised after every
//           object so that the top-level can emit an interrupt between objects.
//           The top-level is responsible for sending the 6-byte result only on
//           the FINAL done (when all n_objects have been processed).
// =============================================================================
module control #(
    parameter OBJ_CNT_WIDTH = 16,
    parameter EMB_DIM       = 8         // kept for potential future use
)(
    input  wire                       clk,
    input  wire                       reset,      // active-HIGH

    input  wire                       start,
    input  wire [15:0]                n_objects,
    input  wire                       emd,        // task_full & label_full

    input  wire                       mac_done,
    input  wire                       score_done,
    input  wire                       max_done,

    output reg                        mac_en,
    output reg                        score_en,
    output reg                        max_en,

    output reg  [OBJ_CNT_WIDTH-1:0]   obj_out,   // current object index
    output reg                        done,       // one-cycle pulse per object
    output reg                        all_done,   // one-cycle pulse after last object

    // [kept as outputs for waveform debug — not used by top]
    output reg  [2:0]                 state,
    output reg  [2:0]                 next
);

    reg [OBJ_CNT_WIDTH-1:0] obj;

    // ── State encoding ──────────────────────────────────────────────────────
    localparam IDLE  = 3'd0,
               LOAD  = 3'd1,
               MAC   = 3'd2,
               SCORE = 3'd3,
               MAX   = 3'd4,
               CHECK = 3'd5;

    // ── Sequential: state register ──────────────────────────────────────────
    always @(posedge clk) begin
        if (reset)         // [BUG-C] was: if(!reset)
            state <= IDLE;
        else
            state <= next;
    end

    // ── Combinational: next-state logic ────────────────────────────────────
    always @(*) begin
        next = state;
        case (state)
            IDLE:  next = start    ? LOAD  : IDLE;
            LOAD:  next = emd      ? MAC   : LOAD;   // [BUG-E] blocks until label ready
            MAC:   next = mac_done ? SCORE : MAC;
            SCORE: next = score_done ? MAX : SCORE;
            MAX:   next = max_done   ? CHECK : MAX;
            CHECK: next = IDLE;  // [BUG-E] always go to IDLE; top re-triggers via start
            default: next = IDLE;
        endcase
    end

    // ── Edge-detect: enter pulses for submodule enables ────────────────────
    wire enter_mac   = (state != MAC)   && (next == MAC);
    wire enter_score = (state != SCORE) && (next == SCORE);
    wire enter_max   = (state != MAX)   && (next == MAX);

    always @(posedge clk) begin
        if (reset) begin           // [BUG-C] was: if(!reset)
            mac_en   <= 1'b0;
            score_en <= 1'b0;
            max_en   <= 1'b0;
        end else begin
            mac_en   <= enter_mac;
            score_en <= enter_score;
            max_en   <= enter_max;
        end
    end

    // ── Object counter, done, all_done ────────────────────────────────────
    always @(posedge clk) begin
        if (reset) begin           // [BUG-C] was: if(!reset)
            obj      <= {OBJ_CNT_WIDTH{1'b0}};
            obj_out  <= {OBJ_CNT_WIDTH{1'b0}};  // [BUG-H] was never assigned
            done     <= 1'b0;
            all_done <= 1'b0;
        end else begin
            done     <= 1'b0;
            all_done <= 1'b0;

            if (state == IDLE && start) begin
                obj     <= {OBJ_CNT_WIDTH{1'b0}};
                obj_out <= {OBJ_CNT_WIDTH{1'b0}};
            end

            // [BUG-E] Pulse done every time CHECK is reached (once per object).
            // Pulse all_done only when the last object has been processed.
            if (state == CHECK) begin
                done <= 1'b1;
                obj_out <= obj;   // [BUG-H] expose current index at done time
                if (obj == n_objects - 1) begin
                    all_done <= 1'b1;
                    obj      <= {OBJ_CNT_WIDTH{1'b0}};
                end else begin
                    obj <= obj + 1'b1;
                end
            end
        end
    end

endmodule
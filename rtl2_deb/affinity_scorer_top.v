`timescale 1ns / 1ps
// =============================================================================
// affinity_scorer_top.v — AXI-Stream Top-Level
// =============================================================================
// DVCon India 2026 - Stage 2B
//
// FIXES applied in this file:
//   [BUG-B] EMB_DIM parameter was never forwarded to submodule instantiations;
//           all submodules were using their default (512). Added #(.EMB_DIM(EMB_DIM))
//           to every submodule instantiation.
//   [BUG-C] All submodules were instantiated with .reset(!axi_reset_n), which
//           produces an active-HIGH signal, but the original submodules used
//           if(!reset) (active-LOW check) → behaviour was inverted.
//           Submodules have been fixed to use if(reset) (active-HIGH), and the
//           top still drives .reset(rst) where rst = !axi_reset_n (active-HIGH).
//           The top's own always blocks use the same active-HIGH rst.
//   [BUG-E] label_full was never cleared between objects, so on the second
//           iteration the LOAD state passed immediately using the stale obj0
//           label buffer. label_full is now cleared when the FSM leaves the
//           LOAD→MAC transition (signalled by mac_en), forcing the FSM to wait
//           for the next label embedding to be streamed in by the testbench.
//   [BUG-E] The testbench (and intended protocol) expects one interrupt per
//           object processed so it can stream the next object between interrupts.
//           The control FSM now exposes a separate all_done pulse (final object)
//           in addition to the per-object done pulse. The top:
//             • fires o_intr on every done (one per object) so the TB can
//               continue streaming the next object
//             • sends the 6-byte output only on all_done (final result)
//   [BUG-A] MAC port was unpacked array — now flat packed bus (fixed in MAC.v).
//           top already drove flat buses so the wiring is unchanged here.
// =============================================================================

module affinity_scorer_top #(
    parameter EMB_DIM = 8
)(
    input  wire        axi_clk,
    input  wire        axi_reset_n,

    // ─── Slave AXI-Stream (input) ────────────────────────────────────────
    input  wire        i_data_valid,
    input  wire [7:0]  i_data,
    output wire        o_data_ready,

    // ─── Master AXI-Stream (output) ──────────────────────────────────────
    output reg         o_data_valid,
    output reg  [7:0]  o_data,
    input  wire        i_data_ready,

    // ─── Interrupt ───────────────────────────────────────────────────────
    output reg         o_intr
);

    // ─── Internal Reset (active-HIGH for all submodules) ─────────────────
    wire rst = !axi_reset_n;

    // ─── Inlined Embedding Buffers ────────────────────────────────────────
    reg signed [7:0]            task_buf  [0:EMB_DIM-1];
    reg signed [7:0]            label_buf [0:EMB_DIM-1];
    reg                         task_full;
    reg                         label_full;

    // Flatten buffers to flat packed buses for MAC
    wire signed [8*EMB_DIM-1:0] task_emb_flat;
    wire signed [8*EMB_DIM-1:0] label_emb_flat;

    genvar gi;
    generate
        for (gi = 0; gi < EMB_DIM; gi = gi + 1) begin : FLATTEN
            assign task_emb_flat [8*gi +: 8] = task_buf[gi];
            assign label_emb_flat[8*gi +: 8] = label_buf[gi];
        end
    endgenerate

    // ─── Input Parser FSM ─────────────────────────────────────────────────
    localparam TYPE_IDLE      = 3'd0;
    localparam TYPE_TASK_EMB  = 3'd1;
    localparam TYPE_LABEL_EMB = 3'd2;
    localparam TYPE_META      = 3'd3;
    localparam TYPE_N_OBJECTS = 3'd4;
    localparam TYPE_START     = 3'd5;

    reg [2:0]               rx_state;
    reg [$clog2(EMB_DIM):0] rx_count;

    reg signed [15:0]       confidence_reg;
    reg signed [15:0]       boost_reg;
    reg                     penalty_reg;
    reg        [15:0]       n_objects_reg;
    reg                     start_reg;      // one-cycle pulse

    assign o_data_ready = 1'b1;

    integer j;

    // mac_en from control: used to clear label_full after LOAD→MAC transition
    wire mac_en;   // forward declaration — assigned by control instance below

    always @(posedge axi_clk) begin
        if (rst) begin
            rx_state       <= TYPE_IDLE;
            rx_count       <= 0;
            confidence_reg <= 16'sd0;
            boost_reg      <= 16'sd0;
            penalty_reg    <= 1'b0;
            n_objects_reg  <= 16'd0;
            start_reg      <= 1'b0;
            task_full      <= 1'b0;
            label_full     <= 1'b0;
            for (j = 0; j < EMB_DIM; j = j + 1) begin
                task_buf[j]  <= 8'sd0;
                label_buf[j] <= 8'sd0;
            end
        end
        else begin
            start_reg <= 1'b0;

            // [BUG-E] Clear label_full as soon as the MAC starts, so the FSM
            // will block in LOAD on the next iteration until the testbench
            // streams in the next object embedding.
            if (mac_en)
                label_full <= 1'b0;

            if (i_data_valid && o_data_ready) begin
                case (rx_state)

                    TYPE_IDLE: begin
                        rx_count <= 0;
                        case (i_data)
                            8'h01: rx_state <= TYPE_TASK_EMB;
                            8'h02: rx_state <= TYPE_LABEL_EMB;
                            8'h03: rx_state <= TYPE_META;
                            8'h04: rx_state <= TYPE_N_OBJECTS;
                            8'h05: begin
                                start_reg <= 1'b1;
                                rx_state  <= TYPE_IDLE;
                            end
                            default: rx_state <= TYPE_IDLE;
                        endcase
                    end

                    TYPE_TASK_EMB: begin
                        task_buf[rx_count] <= $signed(i_data);
                        rx_count           <= rx_count + 1;
                        if (rx_count == EMB_DIM - 1) begin
                            task_full <= 1'b1;
                            rx_state  <= TYPE_IDLE;
                        end
                    end

                    TYPE_LABEL_EMB: begin
                        label_buf[rx_count] <= $signed(i_data);
                        rx_count            <= rx_count + 1;
                        if (rx_count == EMB_DIM - 1) begin
                            label_full <= 1'b1;
                            rx_state   <= TYPE_IDLE;
                        end
                    end

                    TYPE_META: begin
                        rx_count <= rx_count + 1;
                        case (rx_count)
                            0: confidence_reg[15:8] <= $signed(i_data);
                            1: confidence_reg[7:0]  <= i_data;
                            2: boost_reg[15:8]      <= $signed(i_data);
                            3: boost_reg[7:0]       <= i_data;
                            4: begin
                                penalty_reg <= i_data[0];
                                rx_state    <= TYPE_IDLE;
                            end
                        endcase
                    end

                    TYPE_N_OBJECTS: begin
                        rx_count <= rx_count + 1;
                        case (rx_count)
                            0: n_objects_reg[15:8] <= i_data;
                            1: begin
                                n_objects_reg[7:0] <= i_data;
                                rx_state           <= TYPE_IDLE;
                            end
                        endcase
                    end

                    default: rx_state <= TYPE_IDLE;

                endcase
            end
        end
    end

    // ─── Core Compute Submodules ──────────────────────────────────────────
    wire        score_en, max_en;
    wire        mac_done, score_done, max_done;
    wire        done;        // per-object done
    wire        all_done;    // fires only after the last object
    wire [15:0] cur_obj_idx;

    wire signed [31:0] dot_product;
    wire signed [31:0] final_score;
    wire signed [31:0] best_score;
    wire        [15:0] best_object;

    wire emd_signal = task_full & label_full;

    // [BUG-B] Added #(.EMB_DIM(EMB_DIM)) to every submodule instantiation
    // [BUG-C] .reset(rst) — rst is active-HIGH, submodules now expect active-HIGH
    control #(.EMB_DIM(EMB_DIM)) FSM (
        .clk        (axi_clk),
        .reset      (rst),              // [BUG-C] was: .reset(!axi_reset_n) which is same as rst, but submodule logic was inverted
        .start      (start_reg),
        .n_objects  (n_objects_reg),
        .emd        (emd_signal),
        .mac_done   (mac_done),
        .score_done (score_done),
        .max_done   (max_done),
        .mac_en     (mac_en),
        .score_en   (score_en),
        .max_en     (max_en),
        .obj_out    (cur_obj_idx),
        .done       (done),
        .all_done   (all_done)
    );

    MAC #(.EMB_DIM(EMB_DIM)) mac (      // [BUG-B] was: MAC mac (no parameter)
        .clk            (axi_clk),
        .reset          (rst),
        .task_emb_flat  (task_emb_flat),
        .label_emb_flat (label_emb_flat),
        .i_data_valid   (mac_en),
        .dot_product    (dot_product),
        .o_data_valid   (mac_done)
    );

    score Score (                       // no EMB_DIM parameter in score module
        .clk         (axi_clk),
        .reset       (rst),
        .dot         (dot_product),
        .confidence  (confidence_reg),
        .boost       (boost_reg),
        .penalty     (penalty_reg),
        .i_data_valid(mac_done),
        .final_score (final_score),
        .o_data_valid(score_done)
    );

    max_score MS (
        .clk         (axi_clk),
        .reset       (rst),
        .i_data_valid(score_done),
        .final_score (final_score),
        .object_id   (cur_obj_idx),
        .o_data_valid(max_done),
        .best_score  (best_score),
        .best_object (best_object)
    );

    // ─── Output Stream ────────────────────────────────────────────────────
    // 6-byte packet: best_object[15:8], best_object[7:0],
    //                best_score[31:24], best_score[23:16],
    //                best_score[15:8],  best_score[7:0]
    //
    // [BUG-E] o_intr fires on every done (once per object) so the testbench
    //         can stream the next object between interrupts.
    //         The 6-byte result packet is only sent on all_done (final object).

    localparam OUT_IDLE = 2'd0;
    localparam OUT_SEND = 2'd1;

    reg [1:0]  tx_state;
    reg [2:0]  tx_count;
    reg [47:0] tx_buf;

    always @(posedge axi_clk) begin
        if (rst) begin
            tx_state     <= OUT_IDLE;
            tx_count     <= 3'd0;
            tx_buf       <= 48'd0;
            o_data_valid <= 1'b0;
            o_data       <= 8'd0;
            o_intr       <= 1'b0;
        end
        else begin
            // [BUG-E] Pulse intr on every per-object done so TB can stream
            //         the next object in between objects.
            o_intr <= done;

            case (tx_state)

                OUT_IDLE: begin
                    o_data_valid <= 1'b0;
                    // [BUG-E] Only send result packet on all_done (final object).
                    if (all_done) begin
                        tx_buf   <= {best_object, best_score};
                        tx_count <= 3'd0;
                        tx_state <= OUT_SEND;
                    end
                end

                OUT_SEND: begin
                    if (i_data_ready) begin
                        o_data_valid <= 1'b1;
                        o_data       <= tx_buf[47:40];
                        tx_buf       <= {tx_buf[39:0], 8'd0};
                        tx_count     <= tx_count + 1'b1;
                        if (tx_count == 3'd5)
                            tx_state <= OUT_IDLE;
                    end
                    else begin
                        o_data_valid <= 1'b0;
                    end
                end

                default: tx_state <= OUT_IDLE;

            endcase
        end
    end

endmodule
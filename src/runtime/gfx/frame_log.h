// The frame log: every frame's numbers, an event a line, in the file
// RECOMP_FRAME_LOG names (unset: no log, and nothing here costs more than a
// load). Not the perf report's averages over seconds, which hide a frame
// time that swings between 17 and 37 ms: what each frame did, where the
// player was and what was slow, to be opened frame by frame afterwards and
// replayed.
//
// Plain text; a line's first token is its kind; every number is decimal
// unless its name says hex; a reader skips the kinds it does not know.
// "t_s" is seconds since the log began, from one clock for every writer
// (FrameLogSeconds). The kinds:
//
//   # <free text>                        header and notes
//   p <serial> <t_s> <interval_ms> <draws> <passes> <ops>
//        a frame presented (the render thread, at Present): the device's
//        frame serial, ms since the last one, its draws, render passes and
//        recorded ops
//   g <serial> <gpu_busy_ms> <gpu_start_s> <gpu_end_s>
//        the GPU finished frame <serial>: its busy time, with that of the
//        parts a readback cut off before its Present (the work its "q"
//        lists); start and end of its last command buffer, the host's clock
//   n <pass_id> <name with spaces>       a GPU pass name's id, once, before
//                                        its first use
//   q <serial> <pass_id>:<ms> ...        that frame's GPU time, every pass
//                                        in the order it ran (a pass: from
//                                        the one before's end to its own)
//   c <serial> <slot_wait_ms> <record_ms> <rt_note_ms> <snapshot_ms> <end_ms>
//        the render thread's CPU of the frame: blocked waiting for a frame
//        slot; from having the slot to Present without the later waits; and
//        of that, noting draws for the ray scene (an estimate: 1 draw in 32
//        timed, scaled), the ray scene's snapshots, ending its frame
//   r <serial> <tlas_instances> <blas_builds> <captured_instances>
//     <lights_on_screen> <forward_ray_draws>
//        the ray scene's counts of the frame (0 where unknown): instances
//        in the TLAS its effects traced, BLASes built anew (refits of
//        captured draws' not counted), captured draws among the instances,
//        lights the many-lights pass binned on screen, draws that traced
//        rays of their own
//   s <display_s>  /  sg <display_s>     a rendered / generated frame shown
//                                        (the host's clock; 0: dropped)
//        one "s" a presented frame, also where it was not shown (0: the
//        display dropped it, or the window was hidden or gave no drawable).
//        It has no serial and comes when the display reports, 1 to 3 frames
//        after the "p", while a frame with no drawable is written at once:
//        around such a frame the lines may be out of the frames' order, so
//        a shown one is best placed by its time, not by counting
//   w <t_s> <x> <y> <z> <heading> <in_car 0|1> <speed_mps> <cam_x> <cam_y>
//     <cam_z> <cam_heading> <cam_pitch> <fov> <hour> <minute> <weather>
//     <wanted> <interior>                the world, each game frame
//   i <t_s> <lx> <ly> <rx> <ry> <lt> <rt> <buttons_hex> <keys_hex>
//        the pad's and the keyboard's state, when it changes
//   m <t_s> <main_frame_ms> <wait_render_ms>
//        the game's main thread: its frame, its wait for the render thread
//   e <t_s> <text>                       an event ("hitch <ms>", "mark ...")
//
// The host's clock (g, s, sg) is put on t_s by the header line
// "# host clock <host_s> at t_s <t_s>" the backend writes.
//
// Lines gather in a 1 MB buffer and go to the file four times a second
// from a thread of the log's own, never from the thread that wrote them: a
// slow disk does not show up in the frames being measured (lines past a
// full buffer are dropped and counted in a "#" line). A process the system
// kills keeps all but its last quarter second. So does one that leaves
// through _Exit, which runs no exit hook, unless the code that ends it
// calls FrameLogFlush first: the place for that call is before every _Exit
// of a normal quit (the runtime's ExitProcess).

#pragma once

namespace recomp::gfx {

// Whether RECOMP_FRAME_LOG named a file and it opened (the first call of
// any of these opens it; after that one load).
bool FrameLogOn();

// Seconds since the log began (a steady clock); 0 with no log.
double FrameLogSeconds();

// One line (no newline in `fmt`; one inside its text is written as a
// space), from any thread, in the order of the calls. Nothing with no log.
void FrameLogLine(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

// Everything written so far in the file when it returns: to be called on
// the path a process leaves by without running exit's hooks (before
// ExitProcess's _Exit). exit itself runs it. Nothing with no log open.
void FrameLogFlush();

}  // namespace recomp::gfx

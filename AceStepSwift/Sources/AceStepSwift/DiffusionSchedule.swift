/**
 Timestep schedule for DiT diffusion. Mirrors acestep/models/mlx/dit_generate.py get_timestep_schedule.
 */

public enum DiffusionSchedule {
    public static let validShifts: [Double] = [1.0, 2.0, 3.0]

    public static let validTimesteps: [Double] = [
        1.0, 0.9545454545454546, 0.9333333333333333, 0.9, 0.875,
        0.8571428571428571, 0.8333333333333334, 0.7692307692307693, 0.75,
        0.6666666666666666, 0.6428571428571429, 0.625, 0.5454545454545454,
        0.5, 0.4, 0.375, 0.3, 0.25, 0.2222222222222222, 0.125,
    ]

    public static let shiftTimesteps: [Double: [Double]] = [
        1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
        2.0: [
            1.0, 0.9333333333333333, 0.8571428571428571, 0.7692307692307693,
            0.6666666666666666, 0.5454545454545454, 0.4, 0.2222222222222222
        ],
        3.0: [
            1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75,
            0.6428571428571429, 0.5, 0.3
        ],
    ]

    /// Returns timestep schedule (descending, no trailing zero). Matches Python get_timestep_schedule.
    public static func getTimestepSchedule(
        shift: Double = 3.0,
        timesteps: [Double]? = nil,
        inferSteps: Int? = nil
    ) -> [Double] {
        if let ts = timesteps, !ts.isEmpty {
            var list = ts
            while list.last == 0 { list.removeLast() }
            if list.isEmpty {
                return getTimestepSchedule(shift: shift, timesteps: nil, inferSteps: nil)
            }
            let capped = Array(list.prefix(20))
            return capped.map { t in validTimesteps.min(by: { abs($0 - t) < abs($1 - t) })! }
        }

        if let steps = inferSteps, steps > 0 {
            var raw = (0..<steps).map { 1.0 - Double($0) / Double(steps) }
            if shift != 1.0 {
                raw = raw.map { t in shift * t / (1.0 + (shift - 1.0) * t) }
            }
            return raw
        }

        let nearestShift = validShifts.min(by: { abs($0 - shift) < abs($1 - shift) })!
        return shiftTimesteps[nearestShift]!
    }
}

using System;
using System.Diagnostics;
using System.Linq;
using MPI;

class Program
{
    static void Main(string[] args)
    {
        using (new MPI.Environment(ref args))
        {
            var comm = Communicator.world;
            int rank = comm.Rank;
            int size = comm.Size;

            int[] testSizes = { 1_000_000, 10_000_000, 50_000_000 };
            double[,] timings = new double[testSizes.Length, 3];

            foreach (var (totalSize, index) in testSizes.Select((val, idx) => (val, idx)))
            {
                int baseCount = totalSize / size;
                int remainder = totalSize % size;

                int[] counts = Enumerable.Repeat(baseCount, size).ToArray();
                for (int i = 0; i < remainder; i++) counts[i]++;
                int[] displacements = new int[size];
                for (int i = 1; i < size; i++)
                    displacements[i] = displacements[i - 1] + counts[i - 1];

                int[] data = null;
                if (rank == 0)
                {
                    data = new int[totalSize];
                    var random = new Random(42);
                    for (int i = 0; i < totalSize; i++)
                        data[i] = random.Next(1_000_000);

                    Console.WriteLine($"\n[INFO] Processing array of size: {totalSize:N0}");
                }

                int[] localData = new int[counts[rank]];
                comm.ScatterFromFlattened(data, counts, 0, ref localData);

                Stopwatch sw;

                // 1. Minimum
                sw = Stopwatch.StartNew();
                int localMin = localData.Min();
                int globalMin = comm.Reduce(localMin, Operation<int>.Min, 0);
                if (rank == 0)
                {
                    sw.Stop();
                    timings[index, 0] = sw.Elapsed.TotalSeconds;
                }

                // 2. Maximum
                sw = Stopwatch.StartNew();
                int localMax = localData.Max();
                int globalMax = comm.Reduce(localMax, Operation<int>.Max, 0);
                if (rank == 0)
                {
                    sw.Stop();
                    timings[index, 1] = sw.Elapsed.TotalSeconds;
                }

                // 3. Sum
                sw = Stopwatch.StartNew();
                long localSum = localData.Select(x => (long)x).Sum();
                long globalSum = comm.Reduce(localSum, Operation<long>.Add, 0);
                if (rank == 0)
                {
                    sw.Stop();
                    timings[index, 2] = sw.Elapsed.TotalSeconds;
                }
            }

            if (rank == 0)
            {
                Console.WriteLine("\n================== Execution Summary ==================");
                Console.WriteLine("{0,15} | {1,10} | {2,10} | {3,10}", "Array Size", "Min (s)", "Max (s)", "Sum (s)");
                Console.WriteLine(new string('-', 56));

                for (int i = 0; i < testSizes.Length; i++)
                {
                    Console.WriteLine("{0,15:N0} | {1,10:F4} | {2,10:F4} | {3,10:F4}",
                        testSizes[i],
                        timings[i, 0],
                        timings[i, 1],
                        timings[i, 2]);
                }

                Console.WriteLine("=======================================================\n");
            }
        }
    }
}

package org.example;

import org.jtransforms.fft.DoubleFFT_1D;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

class SequencePair {
    final long[] a;
    final long[] b;
    final String method;

    private static long[] negateSeq(long[] seq) {
        if (seq == null) return null;
        long[] res = new long[seq.length];
        for (int i = 0; i < seq.length; i++) res[i] = -seq[i];
        return res;
    }

    private static long[] reverseSeq(long[] seq) {
        if (seq == null) return null;
        long[] reversed = new long[seq.length];
        for (int i = 0; i < seq.length; i++) {
            reversed[i] = seq[seq.length - 1 - i];
        }
        return reversed;
    }

    private static long[] cyclicallyShiftSeq(long[] seq, int shift) {
        if (seq == null || seq.length == 0) return seq;
        int n = seq.length;
        int s = shift % n;
        if (s < 0) s += n;
        if (s == 0) return Arrays.copyOf(seq, n);

        long[] shifted = new long[n];
        System.arraycopy(seq, n - s, shifted, 0, s);
        System.arraycopy(seq, 0, shifted, s, n - s);
        return shifted;
    }

    private static class PairRepresentation implements Comparable<PairRepresentation> {
        long[] sA, sB;

        PairRepresentation(long[] seqA, long[] seqB) {
            this.sA = Arrays.copyOf(seqA, seqA.length);
            this.sB = Arrays.copyOf(seqB, seqB.length);
        }

        @Override
        public int compareTo(PairRepresentation other) {
            int cmp = Arrays.compare(this.sA, other.sA);
            if (cmp != 0) return cmp;
            return Arrays.compare(this.sB, other.sB);
        }
    }

    SequencePair(long[] rawA, long[] rawB, String method) {
        this.method = method;
        if (rawA.length == 0) {
            this.a = rawA;
            this.b = rawB;
            return;
        }

        List<PairRepresentation> equivalentForms = new ArrayList<>();
        int v = rawA.length;

        long[][][] basePairs = {
                {rawA, rawB},
                {rawB, rawA}
        };

        for (long[][] pairContainer : basePairs) {
            long[] pA = pairContainer[0];
            long[] pB = pairContainer[1];

            long[][][] transformations = {
                    {pA, pB},
                    {negateSeq(pA), negateSeq(pB)},
                    {reverseSeq(pA), reverseSeq(pB)},
                    {negateSeq(reverseSeq(pA)), negateSeq(reverseSeq(pB))}
            };

            for (long[][] transformedPairContainer : transformations) {
                long[] transA = transformedPairContainer[0];
                long[] transB = transformedPairContainer[1];

                for (int s = 0; s < v; s++) {
                    equivalentForms.add(new PairRepresentation(
                            cyclicallyShiftSeq(transA, s),
                            cyclicallyShiftSeq(transB, s)
                    ));
                }
            }
        }

        Collections.sort(equivalentForms);
        PairRepresentation canonical = equivalentForms.getFirst();
        this.a = canonical.sA;
        this.b = canonical.sB;
    }


    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        SequencePair that = (SequencePair) o;
        return Arrays.equals(a, that.a) && Arrays.equals(b, that.b);
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(a);
        result = 31 * result + Arrays.hashCode(b);
        return result;
    }

    @Override
    public String toString() {
        return "Pair(method=" + method + ", a=" + Arrays.toString(a) + ", b=" + Arrays.toString(b) + ")";
    }
}


public class Main {
    static final int N_TARGET = 154;
    static final int THREADS = Runtime.getRuntime().availableProcessors();
    static final AtomicBoolean criticalErrorOccurred = new AtomicBoolean(false);
    static final AtomicBoolean solutionFoundGlobal = new AtomicBoolean(false);

    static final int W_CANDIDATES_PER_THREAD_ITERATION = 20_000;
    static final int M_HASH_BITS = 16;
    static final int MAX_BUCKET_SIZE = 75;
    static final long SEARCH_TIME_LIMIT_MS = 24L * 60 * 60 * 1000;
    static final int MAX_UNIQUE_PAIRS_TO_FIND = 100;
    static final int MAX_CHECKS_PER_BUCKET_PAIR = 10_000;

    static final boolean SUM_FILTER_ENABLED = true;
    static final double SUM_FILTER_SQRT_V_MULTIPLIER = 3.0;
    static final boolean PSD_FILTER_ENABLED = true;
    static final double PSD_ADDITIVE_THRESHOLD = 2.0;

    private static final String OUTPUT_FILENAME = "balonin_" + N_TARGET + "_superfast.js";
    private static final ConcurrentHashMap<Integer, DoubleFFT_1D> fftCache = new ConcurrentHashMap<>();
    private static final ThreadLocal<double[]> fftDataBufferThreadLocal = ThreadLocal.withInitial(() -> null);


    public static void main(String[] args) {
        System.out.println("Цель: Найти порядок n = " + N_TARGET + " максимально быстро.");
        System.out.println("Используется потоков: " + THREADS);
        System.out.println("Полная канонизация пар: Включена");
        System.out.println("Фильтр по сумме: " + (SUM_FILTER_ENABLED ? "Включен (|S| <= " + SUM_FILTER_SQRT_V_MULTIPLIER + "*sqrt(v))" : "Выключен"));
        System.out.println("PSD фильтр: " + (PSD_FILTER_ENABLED ? "Включен (порог v + " + PSD_ADDITIVE_THRESHOLD + ")" : "Выключен"));
        System.out.println("Результат будет записан в файл: " + OUTPUT_FILENAME);
        System.out.println("--------------------------------------------------");

        try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(OUTPUT_FILENAME)))) {
            writer.println("// VERSION: " + new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()));
            writer.println("// Поиск для n=" + N_TARGET + " с оптимизацией на скорость нахождения первой пары.");
            writer.println("\nn=" + N_TARGET + "; example(n,0); \n");
            writer.println("\nputs(\"a=[\"+a+\"];\"); puts(\"b=[\"+b+\"];\"); ");
            writer.println("H=twocircul(a,b); {{I=H'*H}} putm(I);");
            writer.println("plotm(H,'XR',140,20);\n");
            writer.println("function example(n, k) {");

            System.out.println("\nИщем уникальные пары для n = " + N_TARGET);
            long startTime = System.currentTimeMillis();

            List<SequencePair> uniquePairsForN = findEulerPairsForN();

            long endTime = System.currentTimeMillis();

            if (!uniquePairsForN.isEmpty()) {
                System.out.printf("+++ Найдено %d уникальных пар для n = %d за %.3f сек +++\n",
                                  uniquePairsForN.size(), N_TARGET, (endTime - startTime) / 1000.0);

                writer.println("    if (n == " + N_TARGET + ") {");
                for (int k_idx = 0; k_idx < uniquePairsForN.size(); k_idx++) {
                    SequencePair pair = uniquePairsForN.get(k_idx);
                    writer.println("        if (k == " + k_idx + ") {");
                    writeArrayToFileJS(writer, pair.a, "a");
                    writeArrayToFileJS(writer, pair.b, "b");
                    writer.println("        }");
                }
                writer.println("    }");
                writer.flush();
            } else {
                System.out.println("--- Решения для n=" + N_TARGET + " не найдены доступными методами/временем ---");
                writer.println("    // No pairs found for n = " + N_TARGET);
            }
            System.out.println("--------------------------------------------------");

            writer.println("}");

        } catch (IOException e) {
            System.err.println("Ошибка записи в файл '" + OUTPUT_FILENAME + "': " + e.getMessage());
            e.printStackTrace();
        }

        System.out.println("Поиск завершен. Результаты записаны в " + OUTPUT_FILENAME);
        if (criticalErrorOccurred.get() || solutionFoundGlobal.get()) {
            System.exit(0);
        }
    }


    static List<SequencePair> findEulerPairsForN() {
        System.out.println("  Запуск эвристического поиска для n=" + Main.N_TARGET + ":");
        System.out.printf("  * Параметры: Лимит времени: %.1fs, До %d пар, PSD фильтр: %s, Сумм.фильтр: %s, Хеш-бит: %d\n",
                          (SEARCH_TIME_LIMIT_MS / 1000.0), MAX_UNIQUE_PAIRS_TO_FIND,
                          (PSD_FILTER_ENABLED ? "ON" : "OFF"),
                          (SUM_FILTER_ENABLED ? "ON" : "OFF"),
                          M_HASH_BITS);


        Set<SequencePair> foundPairsSet = ConcurrentHashMap.newKeySet();
        int v = Main.N_TARGET / 2;
        long targetSumParity = v % 2;
        AtomicLong totalAttempts = new AtomicLong(0);
        AtomicLong passedSumFilter = new AtomicLong(0);
        AtomicLong passedPSDThenStored = new AtomicLong(0);

        long searchStartTime = System.currentTimeMillis();
        long deadline = searchStartTime + SEARCH_TIME_LIMIT_MS;
        final boolean stopAfterFirstFound = MAX_UNIQUE_PAIRS_TO_FIND == 1;

        ExecutorService executor = Executors.newFixedThreadPool(THREADS);
        CountDownLatch latch = new CountDownLatch(THREADS);

        final double maxSumAbsValue = SUM_FILTER_SQRT_V_MULTIPLIER * Math.sqrt(v);

        for (int i = 0; i < THREADS; i++) {
            final int threadId = i;
            executor.submit(() -> {
                Random rand = new Random(System.nanoTime() + threadId);
                Map<Integer, List<SequenceData>> localBucketsA = new HashMap<>();
                Map<Integer, List<SequenceData>> localBucketsB = new HashMap<>();
                try {
                    while (!criticalErrorOccurred.get() && !solutionFoundGlobal.get() && System.currentTimeMillis() < deadline) {
                        localBucketsA.clear();
                        localBucketsB.clear();

                        int candidatesToGenPerIter = W_CANDIDATES_PER_THREAD_ITERATION / THREADS + 1;
                        for (int k_gen = 0; k_gen < candidatesToGenPerIter; k_gen++) {
                            if (criticalErrorOccurred.get() || solutionFoundGlobal.get() || System.currentTimeMillis() >= deadline) break;

                            long[] seqA = randomSeqWithParityAndSumFilter(v, rand, targetSumParity, maxSumAbsValue);
                            if (seqA != null) {
                                passedSumFilter.incrementAndGet();
                                long[] pafA = cyclicAutocorrelation(seqA);
                                if (pafA != null) {
                                    storeInLocalBucket(localBucketsA, calculateHashForPafA(pafA), new SequenceData(seqA, pafA));
                                    passedPSDThenStored.incrementAndGet();
                                }
                            }

                            long[] seqB = randomSeqWithParityAndSumFilter(v, rand, targetSumParity, maxSumAbsValue);
                            if (seqB != null) {
                                passedSumFilter.incrementAndGet();
                                long[] pafB = cyclicAutocorrelation(seqB);
                                if (pafB != null) {
                                    storeInLocalBucket(localBucketsB, calculateComplementaryHashForPafB(pafB), new SequenceData(seqB, pafB));
                                    passedPSDThenStored.incrementAndGet();
                                }
                            }
                            totalAttempts.addAndGet(2);
                        }

                        if (criticalErrorOccurred.get() || solutionFoundGlobal.get() || System.currentTimeMillis() >= deadline) break;

                        Set<Integer> commonHashKeys = new HashSet<>(localBucketsA.keySet());
                        commonHashKeys.retainAll(localBucketsB.keySet());

                        for (int bucketKey : commonHashKeys) {
                            if (criticalErrorOccurred.get() || solutionFoundGlobal.get() || System.currentTimeMillis() >= deadline) break;
                            checkAndAddPairs(bucketKey, v, localBucketsA, localBucketsB, foundPairsSet, deadline);
                        }
                    }
                } catch (OutOfMemoryError oom) {
                    criticalErrorOccurred.set(true);
                    System.err.printf("\nOOM в потоке TreeSearch %d для n=%d. Поиск остановлен.\n", threadId, Main.N_TARGET);
                } catch (Exception e) {
                    if (!criticalErrorOccurred.get() && !solutionFoundGlobal.get()) {
                        System.err.printf("\nОшибка в потоке TreeSearch %d: %s\n", threadId, e.getMessage());
                        e.printStackTrace();
                    }
                } finally {
                    latch.countDown();
                }
            });
        }
        try {
            while (latch.getCount() > 0 && !solutionFoundGlobal.get() && !criticalErrorOccurred.get() && System.currentTimeMillis() < deadline) {
                latch.await(200, TimeUnit.MILLISECONDS);
            }
            if (solutionFoundGlobal.get() && latch.getCount() > 0) {
                latch.await(1, TimeUnit.SECONDS);
            }
        } catch (InterruptedException e) {
            System.err.println("\nОсновной поток прерван во время ожидания.");
            Thread.currentThread().interrupt();
            criticalErrorOccurred.set(true);
        } finally {
            if (solutionFoundGlobal.get() || criticalErrorOccurred.get() || System.currentTimeMillis() >= deadline) {
                executor.shutdownNow();
            } else {
                executor.shutdown();
            }
        }

        try {
            if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
                System.err.println("\nПотоки TreeSearch не завершились вовремя после shutdown.");
            }
        } catch (InterruptedException e) {
            System.err.println("\nПрервано ожидание окончательного завершения потоков TreeSearch.");
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }

        long duration = System.currentTimeMillis() - searchStartTime;
        System.out.printf("  Всего сгенерировано попыток: %.2fM. Прошло сумм.фильтр: %d. Сохранено в бакеты (после PSD): %d.\n",
                          totalAttempts.get() / 1_000_000.0, passedSumFilter.get(), passedPSDThenStored.get());
        System.out.printf("  Затрачено времени: %.2fs.", duration / 1000.0);


        if (System.currentTimeMillis() >= deadline && !solutionFoundGlobal.get() && !criticalErrorOccurred.get()) {
            System.out.print(" [Таймаут]");
        }
        if (criticalErrorOccurred.get()) {
            System.out.print(" [Ошибка OOM или Прерывание]");
        }
        System.out.println();

        return new ArrayList<>(foundPairsSet);
    }

    private static void checkAndAddPairs(int bucketKey, int v_len,
                                         Map<Integer, List<SequenceData>> bucketsA,
                                         Map<Integer, List<SequenceData>> bucketsB,
                                         Set<SequencePair> globalResultsSet,
                                         long deadline
                                        ) {
        List<SequenceData> listA = bucketsA.get(bucketKey);
        List<SequenceData> listB = bucketsB.get(bucketKey);
        if (listA == null || listB == null || listA.isEmpty() || listB.isEmpty()) return;

        long pairsCheckedInBucket = 0;

        for (SequenceData dataA : listA) {
            if (criticalErrorOccurred.get() || solutionFoundGlobal.get() || System.currentTimeMillis() >= deadline) break;
            long[] pafA = dataA.paf;

            for (SequenceData dataB : listB) {
                if (criticalErrorOccurred.get() || solutionFoundGlobal.get() || System.currentTimeMillis() >= deadline) break;

                pairsCheckedInBucket++;
                if (MAX_CHECKS_PER_BUCKET_PAIR > 0 && pairsCheckedInBucket > MAX_CHECKS_PER_BUCKET_PAIR) {
                    return;
                }

                long[] pafB = dataB.paf;
                boolean match = true;
                for (int k = 1; k < v_len; k++) {
                    if (pafA[k] + pafB[k] != -2) {
                        match = false;
                        break;
                    }
                }

                if (match) {
                    boolean added = globalResultsSet.add(new SequencePair(dataA.sequence, dataB.sequence, "HeuristicSearch"));
                    if (added) {
                        solutionFoundGlobal.set(true);
                        return;
                    }
                    if (added && MAX_UNIQUE_PAIRS_TO_FIND > 0 && !globalResultsSet.isEmpty()) {
                        solutionFoundGlobal.set(true);
                        return;
                    }
                }
            }
        }
    }


    private static void storeInLocalBucket(Map<Integer, List<SequenceData>> buckets, int hash, SequenceData data) {
        List<SequenceData> list = buckets.computeIfAbsent(hash, k -> new ArrayList<>());
        if (MAX_BUCKET_SIZE == 0 || list.size() < MAX_BUCKET_SIZE) {
            list.add(data);
        }
    }

    static long[] cyclicAutocorrelation(long[] seq) {
        int v_len = seq.length;
        if (v_len == 0) return null;

        try {
            DoubleFFT_1D fft = fftCache.computeIfAbsent(v_len, DoubleFFT_1D::new);
            double[] data = fftDataBufferThreadLocal.get();

            int requiredSize = 2 * v_len;
            if (data == null || data.length < requiredSize) {
                int bufferSize = Integer.highestOneBit(requiredSize - 1) << 1;
                if (bufferSize < requiredSize) bufferSize <<= 1;
                if (bufferSize == 0) bufferSize = Math.max(256, requiredSize);
                data = new double[bufferSize];
                fftDataBufferThreadLocal.set(data);
            }
            Arrays.fill(data, 0, Math.min(data.length, 2 * v_len + 2), 0.0);


            for (int i = 0; i < v_len; i++) {
                data[2 * i] = seq[i];
                data[2 * i + 1] = 0;
            }
            if (data.length > 2*v_len) {
                Arrays.fill(data, 2 * v_len, data.length, 0.0);
            }

            fft.complexForward(data);

            for (int i = 0; i < v_len; i++) {
                double re = data[2 * i];
                double im = data[2 * i + 1];
                data[2 * i] = re * re + im * im;
                data[2 * i + 1] = 0;
            }

            if (PSD_FILTER_ENABLED) {
                double psdValueThreshold = (double) v_len + PSD_ADDITIVE_THRESHOLD;
                for (int i = 1; i < v_len; i++) {
                    if (data[2 * i] > psdValueThreshold) {
                        return null;
                    }
                }
            }

            if (data.length > 2*v_len) {
                Arrays.fill(data, 2 * v_len, data.length, 0.0);
            }

            fft.complexInverse(data, true);

            long[] result = new long[v_len];
            for (int i = 0; i < v_len; i++) {
                result[i] = Math.round(data[2 * i]);
            }
            return result;
        } catch (Exception e) {
            return null;
        } catch (OutOfMemoryError oom) {
            criticalErrorOccurred.set(true);
            System.err.println("\nOOM при вычислении АКФ для N=" + v_len);
            throw oom;
        }
    }


    static void writeArrayToFileJS(PrintWriter writer, long[] arr, String symbol) {
        if (arr == null) {
            writer.println("            " + symbol + " = null;");
            return;
        }
        writer.print("            " + symbol + " = [");
        writer.print(Arrays.stream(arr).mapToObj(String::valueOf).collect(Collectors.joining(",")));
        writer.println("];");
    }


    record SequenceData(long[] sequence, long[] paf) {}

    static long[] randomSeqWithParityAndSumFilter(int length, Random rand, long targetSumParity, double maxSumAbsoluteValue) {
        long[] seq = new long[length];
        long currentSum = 0;
        for (int i = 0; i < length; i++) {
            seq[i] = rand.nextBoolean() ? 1L : -1L;
            currentSum += seq[i];
        }

        if (SUM_FILTER_ENABLED && Math.abs(currentSum) > maxSumAbsoluteValue) {
            return null;
        }

        long currentParity = Math.abs(currentSum % 2);
        if (currentParity != targetSumParity) {
            if (length > 0) {
                int flipIndex = rand.nextInt(length);
                long newSum = currentSum - 2 * seq[flipIndex];
                if (SUM_FILTER_ENABLED && Math.abs(newSum) > maxSumAbsoluteValue) {
                    return null;
                }
                seq[flipIndex] *= -1L;
            } else {
                return seq;
            }
        }
        return seq;
    }

    static int calculateHashForPafA(long[] paf) {
        int hash = 0;
        int limit = Math.min(M_HASH_BITS + 1, paf.length);
        for (int i = 1; i < limit; i++) {
            if (paf[i] > 0) {
                hash |= (1 << (i - 1));
            }
        }
        return hash;
    }

    static int calculateComplementaryHashForPafB(long[] pafB) {
        int hash = 0;
        int limit = Math.min(M_HASH_BITS + 1, pafB.length);
        for (int i = 1; i < limit; i++) {
            if ((-2 - pafB[i]) > 0) {
                hash |= (1 << (i - 1));
            }
        }
        return hash;
    }
}
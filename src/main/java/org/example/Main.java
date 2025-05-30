package org.example;

import org.jtransforms.fft.DoubleFFT_1D;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import java.util.stream.Collectors;

class SequencePair {
    final long[] a;
    final long[] b;
    final String method;
    final int n;

    private static long[] negateSeqInternal(long[] seq) {
        if (seq == null) return null;
        long[] res = new long[seq.length];
        for (int i = 0; i < seq.length; i++) res[i] = -seq[i];
        return res;
    }

    SequencePair(long[] seqA, long[] seqB, String method, int n) {
        this.method = method;
        this.n = n;

        long[] currentA = Arrays.copyOf(seqA, seqA.length);
        long[] currentB = Arrays.copyOf(seqB, seqB.length);

        if (Arrays.compare(currentA, currentB) > 0) {
            long[] temp = currentA;
            currentA = currentB;
            currentB = temp;
        }

        long[] negatedA = negateSeqInternal(currentA);
        if (Arrays.compare(currentA, negatedA) > 0) {
            this.a = negatedA;
            this.b = negateSeqInternal(currentB);
        } else {
            this.a = currentA;
            this.b = currentB;
        }
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
        return "Pair(n=" + n + ", method=" + method + ", a=" + Arrays.toString(a) + ", b=" + Arrays.toString(b) + ")";
    }
}

public class Main {
    static final int N_MIN = 50;
    static final int N_MAX = 50;
    static final int THREADS = Runtime.getRuntime().availableProcessors();
    static final AtomicBoolean criticalErrorOccurred = new AtomicBoolean(false);

    static final int W_FALLBACK = 15_000;
    static final int M_BITS_FALLBACK = 14;
    static final int MAX_BUCKET_SIZE_FALLBACK = 50;
    static final long FALLBACK_TIME_LIMIT_MS = 24L * 60 * 60 * 1000; // 24 часа
    static final int MAX_UNIQUE_PAIRS_PER_N = 1000;
    static final int MAX_CHECKS_PER_BUCKET_PAIR = 8000;

    static final boolean PSD_FILTER_ENABLED = true;
    static final double PSD_THRESHOLD_ADDITIVE = 2.0;

    private static final String OUTPUT_FILENAME = "balonin_" + N_MIN + "_fast.js";
    private static final ConcurrentHashMap<Integer, DoubleFFT_1D> fftCache = new ConcurrentHashMap<>();
    private static final ThreadLocal<double[]> dataBufferThreadLocal = new ThreadLocal<>();

    private static int calculateK1(int v) {
        return v / 2 + 1;
    }

    public static void main(String[] args) {
        System.out.println("Цель: Найти порядок n = " + N_MIN + " как можно быстрее.");
        System.out.println("Используется потоков: " + THREADS);
        System.out.println("Остановимся после нахождения " + MAX_UNIQUE_PAIRS_PER_N + " уникальной(-ых) пары (пар) для каждого N.");
        System.out.println("PSD фильтр: " + (PSD_FILTER_ENABLED ? "Включен" : "Выключен"));
        System.out.println("Результат будет записан в файл: " + OUTPUT_FILENAME);
        System.out.println("--------------------------------------------------");

        try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(OUTPUT_FILENAME)))) {
            writer.println("// VERSION: " + new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()));
            writer.println("// Поиск для n=" + N_MIN + (N_MAX > N_MIN ? " до n=" + N_MAX : ""));
            writer.println("\n// n=" + N_MIN +"; example(n,0); \n");
            writer.println("\nputs(\"a=[\"+a+\"];\"); puts(\"b=[\"+b+\"];\"); ");
            writer.println("H=twocircul(a,b); {{I=H'*H}} putm(I);");
            writer.println("plotm(H,'XR',140,20);\n");
            writer.println("function example(n, k) {");
            writer.flush();

            boolean firstNBlockWritten = true;

            for (int n_current = N_MIN; n_current <= N_MAX; n_current += 4) {
                if (criticalErrorOccurred.get()) {
                    System.out.println("Критическая ошибка. Поиск прерван.");
                    break;
                }

                System.out.println("\nИщем уникальные пары для n = " + n_current);
                long startTime = System.currentTimeMillis();

                AtomicInteger jsKIndex = new AtomicInteger(0);
                AtomicBoolean jsNBlockHeaderWritten = new AtomicBoolean(false);

                final boolean isFirstNOverall = firstNBlockWritten;
                int finalN_current = n_current;
                Consumer<SequencePair> jsPairWriter = (pair) -> {
                    synchronized (writer) {
                        if (!jsNBlockHeaderWritten.getAndSet(true)) {
                            if (!isFirstNOverall) {
                                writer.print("else ");
                            }
                            writer.println("    if (n == " + finalN_current + ") {");
                        }
                        writer.println("        if (k == " + jsKIndex.getAndIncrement() + ") {");
                        writeArrayToFileJS(writer, pair.a, "a");
                        writeArrayToFileJS(writer, pair.b, "b");
                        writer.println("        }");
                        writer.flush();
                    }
                };

                List<SequencePair> foundPairsForN = findAllPairsDispatcher(n_current, jsPairWriter);

                long endTime = System.currentTimeMillis();

                synchronized (writer) {
                    if (jsNBlockHeaderWritten.get()) {
                        writer.println("    }");
                        writer.flush();
                        firstNBlockWritten = false;
                    }
                }

                if (!foundPairsForN.isEmpty()) {
                    System.out.printf("+++ Всего найдено %d уникальных пар для n = %d за %.3f сек +++\n",
                                      foundPairsForN.size(), n_current, (endTime - startTime) / 1000.0);
                } else {
                    System.out.println("--- Решения для n=" + n_current + " не найдены за отведенное время/попытки ---");
                }
                System.out.println("--------------------------------------------------");

                if (MAX_UNIQUE_PAIRS_PER_N > 0 && !foundPairsForN.isEmpty() && foundPairsForN.size() >= MAX_UNIQUE_PAIRS_PER_N) {
                    System.out.println("Достигнут лимит пар (" + foundPairsForN.size() + ") для n=" + n_current + ". Переход к следующему N (если есть) или завершение.");
                    if (n_current == N_MAX) {
                        System.out.println("Цель достигнута для максимального N. Завершение.");
                        break;
                    }
                }
            }
            writer.println();
            writer.println("}");
            writer.flush();

        } catch (IOException e) {
            System.err.println("Ошибка записи в файл '" + OUTPUT_FILENAME + "': " + e.getMessage());
            e.printStackTrace();
        }

        System.out.println("Поиск завершен. Результаты записаны в " + OUTPUT_FILENAME);
        if (THREADS > 1 || criticalErrorOccurred.get()) {
            System.exit(criticalErrorOccurred.get() ? 1 : 0);
        }
    }

    static List<SequencePair> findAllPairsDispatcher(int n, Consumer<SequencePair> onPairFoundCallback) {
        System.out.println("  Запуск эвристического поиска для n=" + n + ":");
        System.out.print("  * Поиск (лимит времени: " + (FALLBACK_TIME_LIMIT_MS / 1000.0) + "s, до " + MAX_UNIQUE_PAIRS_PER_N + " пар)... ");

        List<SequencePair> foundPairs = findPairsByRandomizedBucketing(n, onPairFoundCallback);
        System.out.println("Итого найдено для n=" + n + ": " + foundPairs.size());
        return foundPairs;
    }

    static List<SequencePair> findPairsByRandomizedBucketing(int n, Consumer<SequencePair> onPairFoundCallback) {
        Set<SequencePair> foundPairsSet = ConcurrentHashMap.newKeySet();
        int v = n / 2;
        if (v <= 0) {
            System.err.println("Некорректное значение v=" + v + " для n=" + n);
            return new ArrayList<>();
        }
        int k1_target = calculateK1(v);

        AtomicLong totalAttempts = new AtomicLong(0);
        AtomicLong totalComparisons = new AtomicLong(0);
        long startTime = System.currentTimeMillis();
        long deadline = startTime + FALLBACK_TIME_LIMIT_MS;

        final boolean unlimitedPairs = MAX_UNIQUE_PAIRS_PER_N <= 0;

        ExecutorService executor = Executors.newFixedThreadPool(THREADS);
        CountDownLatch latch = new CountDownLatch(THREADS);

        for (int i = 0; i < THREADS; i++) {
            int threadId = i;
            executor.submit(() -> {
                Random rand = new Random(System.nanoTime() + threadId);
                Map<Integer, List<SequenceData>> bucketsA_local = new HashMap<>();
                Map<Integer, List<SequenceData>> bucketsB_local = new HashMap<>();
                try {
                    while (!criticalErrorOccurred.get() && System.currentTimeMillis() < deadline &&
                           (unlimitedPairs || foundPairsSet.size() < MAX_UNIQUE_PAIRS_PER_N)) {

                        bucketsA_local.clear();
                        bucketsB_local.clear();

                        int candidatesToGen = W_FALLBACK / THREADS + 1;
                        for (int k_gen = 0; k_gen < candidatesToGen; k_gen++) {
                            if (criticalErrorOccurred.get() || System.currentTimeMillis() >= deadline ||
                                (!unlimitedPairs && foundPairsSet.size() >= MAX_UNIQUE_PAIRS_PER_N)) break;

                            long[] seq = generateSequenceWithK1(v, k1_target, rand);
                            long[] paf = cyclicAutocorrelation(seq);

                            if (paf != null) {
                                SequenceData data = new SequenceData(seq, paf);
                                storeSequenceLocal(bucketsA_local, calculateHashN1(paf), data);
                                storeSequenceLocal(bucketsB_local, calculateHashN2(paf), data);
                            }
                            totalAttempts.incrementAndGet();
                        }

                        if (criticalErrorOccurred.get() || System.currentTimeMillis() >= deadline ||
                            (!unlimitedPairs && foundPairsSet.size() >= MAX_UNIQUE_PAIRS_PER_N)) break;

                        Set<Integer> commonKeys = new HashSet<>(bucketsA_local.keySet());
                        commonKeys.retainAll(bucketsB_local.keySet());

                        for (int key : commonKeys) {
                            if (criticalErrorOccurred.get() || System.currentTimeMillis() >= deadline ||
                                (!unlimitedPairs && foundPairsSet.size() >= MAX_UNIQUE_PAIRS_PER_N)) break;

                            List<SequenceData> listA = bucketsA_local.get(key);
                            List<SequenceData> listB = bucketsB_local.get(key);

                            if (listA == null || listB == null) continue;

                            int checksInBucketPair = 0;

                            for (SequenceData dataA : listA) {
                                if (criticalErrorOccurred.get() || System.currentTimeMillis() >= deadline ||
                                    (!unlimitedPairs && foundPairsSet.size() >= MAX_UNIQUE_PAIRS_PER_N)) break;

                                for (SequenceData dataB : listB) {
                                    if (criticalErrorOccurred.get() || System.currentTimeMillis() >= deadline ||
                                        (!unlimitedPairs && foundPairsSet.size() >= MAX_UNIQUE_PAIRS_PER_N)) break;

                                    totalComparisons.incrementAndGet();
                                    if (checkEulerPairCondition(dataA.paf(), dataB.paf(), v)) {
                                        SequencePair foundPair = new SequencePair(dataA.sequence(), dataB.sequence(), "BucketSearch", n);
                                        boolean added = foundPairsSet.add(foundPair);
                                        if (added) {
                                            System.out.printf("\n[n=%d, Thread %d] +++ Найдена пара! Всего для n=%d найдено: %d. Попыток: %.2fM, Сравнений: %.1fM. PAF_A[0]=%d, PAF_B[0]=%d\n",
                                                              n, threadId, n, foundPairsSet.size(),
                                                              totalAttempts.get() / 1_000_000.0, totalComparisons.get() / 1_000_000.0,
                                                              dataA.paf()[0], dataB.paf()[0]);

                                            onPairFoundCallback.accept(foundPair);

                                            if (!unlimitedPairs && foundPairsSet.size() >= MAX_UNIQUE_PAIRS_PER_N) {
                                                break;
                                            }
                                        }
                                    }
                                    checksInBucketPair++;
                                    if (MAX_CHECKS_PER_BUCKET_PAIR > 0 && checksInBucketPair >= MAX_CHECKS_PER_BUCKET_PAIR) {
                                        break;
                                    }
                                }
                                if ( (MAX_CHECKS_PER_BUCKET_PAIR > 0 && checksInBucketPair >= MAX_CHECKS_PER_BUCKET_PAIR) ||
                                     (!unlimitedPairs && foundPairsSet.size() >= MAX_UNIQUE_PAIRS_PER_N) ) {
                                    break;
                                }
                            }
                        }
                    }
                } catch (OutOfMemoryError oom) {
                    criticalErrorOccurred.set(true);
                    System.err.println("\nOOM в потоке " + threadId + " для n=" + n + ". Поиск для n остановлен.");
                } catch (Exception e) {
                    System.err.println("\nОшибка в потоке " + threadId + " для n=" + n + ": " + e);
                    e.printStackTrace();
                } finally {
                    latch.countDown();
                }
            });
        }

        try {
            while (latch.getCount() > 0 && System.currentTimeMillis() < deadline &&
                   !criticalErrorOccurred.get() && (unlimitedPairs || foundPairsSet.size() < MAX_UNIQUE_PAIRS_PER_N)) {
                latch.await(200, TimeUnit.MILLISECONDS);
            }
        } catch (InterruptedException e) {
            System.err.println("\nОсновной поток прерван во время ожидания для n=" + n);
            Thread.currentThread().interrupt();
            criticalErrorOccurred.set(true);
        }

        if (System.currentTimeMillis() >= deadline || criticalErrorOccurred.get() || (!unlimitedPairs && foundPairsSet.size() >= MAX_UNIQUE_PAIRS_PER_N)) {
            executor.shutdown();
            try {
                if (!executor.awaitTermination(1, TimeUnit.SECONDS)) {
                    List<Runnable> droppedTasks = executor.shutdownNow();
                    if (!droppedTasks.isEmpty()) {
                        System.err.println("Для n=" + n +": Задачи, которые не были выполнены после shutdownNow: " + droppedTasks.size());
                    }
                    if (!executor.awaitTermination(1, TimeUnit.SECONDS)) {
                        System.err.println("Для n=" + n +": Пул потоков не завершился корректно после shutdownNow.");
                    }
                }
            } catch (InterruptedException ie) {
                executor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        } else {
            executor.shutdownNow();
        }


        long duration = System.currentTimeMillis() - startTime;
        System.out.printf("  Статистика для n=%d: Попыток генерации: %.2fM. Сравнений PAF: %.2fM. Время: %.2fs.",
                          n,
                          totalAttempts.get() / 1_000_000.0,
                          totalComparisons.get() / 1_000_000.0,
                          duration / 1000.0);

        if (System.currentTimeMillis() >= deadline && !criticalErrorOccurred.get() && (unlimitedPairs || foundPairsSet.size() < MAX_UNIQUE_PAIRS_PER_N)) {
            System.out.print(" [Таймаут]");
        }
        if (criticalErrorOccurred.get()) {
            System.out.print(" [Ошибка OOM или Прерывание]");
        }
        System.out.println();

        return new ArrayList<>(foundPairsSet);
    }


    private static boolean checkEulerPairCondition(long[] pafA, long[] pafB, int v) {
        if (pafA == null || pafB == null || pafA.length != v || pafB.length != v) {
            return false;
        }

        if (pafA[0] != v || pafB[0] != v) {
            return false;
        }

        for (int k = 1; k < v; k++) {
            if (pafA[k] + pafB[k] != -2) {
                return false;
            }
        }
        return true;
    }


    private static void storeSequenceLocal(Map<Integer, List<SequenceData>> buckets, int hash, SequenceData data) {
        List<SequenceData> list = buckets.computeIfAbsent(hash, k -> new ArrayList<>());
        if (MAX_BUCKET_SIZE_FALLBACK == 0 || list.size() < MAX_BUCKET_SIZE_FALLBACK) {
            list.add(data);
        }
    }

    static long[] cyclicAutocorrelation(long[] seq) {
        int v_len = seq.length;
        if (v_len == 0) return null;

        try {
            DoubleFFT_1D fft = fftCache.computeIfAbsent(v_len, DoubleFFT_1D::new);
            double[] data = dataBufferThreadLocal.get();
            final int requiredFftArraySize = 2 * v_len;

            if (data == null || data.length < requiredFftArraySize) {
                int newBufferLength;
                if (requiredFftArraySize <= 0) {
                    newBufferLength = 256;
                } else {
                    newBufferLength = Integer.highestOneBit(requiredFftArraySize -1) << 1;
                    if (newBufferLength == 0 && requiredFftArraySize > 0) {
                        newBufferLength = 2;
                    } else if (newBufferLength < requiredFftArraySize) {
                        newBufferLength = Integer.highestOneBit(requiredFftArraySize) << 1;
                        if (newBufferLength < requiredFftArraySize) newBufferLength = requiredFftArraySize;
                    }
                }
                newBufferLength = Math.max(newBufferLength, requiredFftArraySize);
                newBufferLength = Math.max(newBufferLength, 256);

                data = new double[newBufferLength];
                dataBufferThreadLocal.set(data);
            } else {
                Arrays.fill(data, 0, Math.min(data.length, requiredFftArraySize * 2), 0.0);
            }


            for (int i = 0; i < v_len; i++) {
                data[2 * i] = seq[i];
            }

            fft.complexForward(data);

            for (int i = 0; i < v_len; i++) {
                double re = data[2 * i];
                double im = data[2 * i + 1];
                data[2 * i] = re * re + im * im;
                data[2 * i + 1] = 0.0;
            }

            if (PSD_FILTER_ENABLED) {
                double psdThreshold = (double)v_len * 2.0 + PSD_THRESHOLD_ADDITIVE;
                for (int i = 1; i < v_len; i++) {
                    if (data[2 * i] > psdThreshold) {
                        return null;
                    }
                }
            }

            if (data.length > requiredFftArraySize) {
                Arrays.fill(data, requiredFftArraySize, data.length, 0.0);
            }

            fft.complexInverse(data, true);

            long[] result = new long[v_len];
            for (int i = 0; i < v_len; i++) {
                result[i] = Math.round(data[2 * i]);
            }
            return result;
        } catch (Exception e) {
            System.err.println("Ошибка при вычислении АКФ для v_len=" + v_len + ": " + e.getMessage() + ". Seq length: " + (seq != null ? seq.length : "null"));
            return null;
        } catch (OutOfMemoryError oom) {
            criticalErrorOccurred.set(true);
            System.err.println("\nOOM при вычислении АКФ для v_len=" + v_len + ". Последовательность отброшена. Поиск будет остановлен.");
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

    static long[] generateSequenceWithK1(int length, int k1, Random rand) {
        if (k1 < 0 || k1 > length) {
            throw new IllegalArgumentException("k1 ("+ k1 +") должно быть в диапазоне [0, " + length + "]");
        }
        long[] seq = new long[length];
        Arrays.fill(seq, 1L);

        if (k1 == length) {
            Arrays.fill(seq, -1L);
            return seq;
        }
        if (k1 == 0) {
            return seq;
        }

        List<Integer> indices = new ArrayList<>(length);
        for (int i = 0; i < length; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, rand);

        for (int i = 0; i < k1; i++) {
            seq[indices.get(i)] = -1L;
        }
        return seq;
    }

    static int calculateHashN1(long[] paf) {
        if (paf == null) return 0;
        int hash = 0;
        int limit = Math.min(M_BITS_FALLBACK + 1, paf.length);
        for (int i = 1; i < limit; i++) {
            if (paf[i] > 0) {
                hash |= (1 << (i - 1));
            }
        }
        return hash;
    }

    static int calculateHashN2(long[] paf) {
        if (paf == null) return 0;
        int hash = 0;
        int limit = Math.min(M_BITS_FALLBACK + 1, paf.length);
        for (int i = 1; i < limit; i++) {
            if (paf[i] < -2) {
                hash |= (1 << (i - 1));
            }
        }
        return hash;
    }
}
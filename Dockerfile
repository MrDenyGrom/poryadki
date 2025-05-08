# Этап сборки (остается без изменений)
FROM maven:3.9-eclipse-temurin-22 AS builder

WORKDIR /app

COPY pom.xml .
# Если есть зависимости в pom.xml, лучше сначала скачать их,
# чтобы использовать кеш Docker при изменении только исходного кода.
# RUN mvn dependency:go-offline -B
COPY src ./src

RUN mvn package -B

# Этап запуска
FROM eclipse-temurin:22-jre-jammy AS runner

WORKDIR /app

# Копируем собранный JAR из этапа сборки
COPY --from=builder /app/target/euler-bicircles-1.0-SNAPSHOT.jar ./app.jar

ENTRYPOINT ["java", "-Xmx16g", "-jar", "/app/app.jar"]
FROM maven:3.9-eclipse-temurin-22 AS builder

WORKDIR /app

COPY pom.xml .

COPY src ./src

RUN mvn package -B

FROM eclipse-temurin:22-jre-jammy AS runner

WORKDIR /app

COPY --from=builder /app/target/euler-bicircles-1.0-SNAPSHOT.jar ./app.jar

ENTRYPOINT ["java", "-Xmx16g", "-jar", "/app/app.jar"]
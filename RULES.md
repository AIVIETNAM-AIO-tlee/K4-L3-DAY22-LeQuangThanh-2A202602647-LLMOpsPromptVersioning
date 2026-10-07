# Quy định làm bài — Day 22: LangSmith + Prompt Versioning

## 1. Hình thức

- **Bài cá nhân.** Mỗi học viên tự làm và tự nộp 1 repo đặt tên theo [SUBMISSION.md](SUBMISSION.md).
- Được trao đổi ý tưởng, cách debug với bạn khác; **không** dùng chung code, prompt, log hay ảnh evidence.

## 2. Sử dụng AI

- **Được phép** dùng AI (ChatGPT, Claude, Copilot, …) để giải thích khái niệm, gợi ý, debug.
- Bạn phải **hiểu và giải thích được** mọi dòng code đã nộp. Coach có thể hỏi trực tiếp; không giải thích được phần nào có thể không được tính điểm phần đó.
- Không dùng validator có sẵn từ Guardrails Hub thay cho phần tự implement (xem [RUBRIC.md](RUBRIC.md)).

## 3. Hợp tác và sao chép

- Code, log, ảnh chụp màn hình phải do chính bạn tạo ra từ môi trường của bạn. Tên prompt trên Hub phải là tên riêng của bạn.
- Bài giống nhau bất thường (code, log, evidence) → tất cả các bài liên quan nhận **0 điểm** phần bị trùng và bị báo cáo lên ban điều phối.
- Evidence giả mạo hoặc chỉnh sửa (ảnh/ log không khớp với code) → 0 điểm cả bài.

## 4. Deadline và nộp muộn

- Deadline: **23:59 ngày học lab (08/10/2026, GMT+7)**, trừ khi key coach thông báo khác trong vòng 48 giờ sau buổi lab.
- Thời điểm nộp = thời điểm của **commit cuối cùng** trên repo và thời điểm nộp link trên cổng khóa học.
- Nộp muộn bị trừ điểm theo quy định chung của khóa học do key coach công bố.

## 5. Sửa bài sau deadline

- Commit sau deadline **không** được chấm; bài được chấm theo commit cuối cùng trước deadline.
- Nếu cần sửa sau deadline (ví dụ bổ sung evidence), phải được key coach đồng ý và tính như nộp muộn.
- Không force-push / viết lại lịch sử commit để thay đổi mốc thời gian.

## 6. Bảo mật API key và dữ liệu

- **Không bao giờ commit `.env`** hay dán API key (OpenAI, Gemini, Anthropic, OpenRouter, LangSmith) vào mã nguồn, log hoặc ảnh chụp màn hình. Vi phạm: **trừ 10 điểm tự động**.
- Nếu lỡ commit key: **thu hồi (revoke) key ngay** trên trang của nhà cung cấp, tạo key mới. Xoá file khỏi commit mới không đủ vì key vẫn nằm trong lịch sử git.
- Che (blur) các thông tin nhạy cảm (API key, email cá nhân, org-id nếu không muốn chia sẻ) trong ảnh chụp màn hình.
- Chỉ dùng dữ liệu trong `data/` và dữ liệu PII **giả** cho test case Guardrails; không dùng thông tin cá nhân thật.
